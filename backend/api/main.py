import sys
import os
import uuid
import time
import json
import re
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime

from src.full_rag import FullRAGSystem
from src.user_docs_service import UserDocsService
from src.agent import (
    route_chat_tools,
    detect_smalltalk,
    chat_response,
    merge_tool_answer,
    execute_tool,
    is_tool_relevant,
    detect_explicit_tool,
    _DEFINITION_QUESTION_RE,
)
from src.agents.notice_agent import LegalNoticeAgent
from src.agents.evidence_agent import EvidenceChecklistAgent
from src.agents.rti_agent import RTIApplicationAgent
from src.agents.deadline_agent import DeadlineTrackerAgent
from src.agents.strategy_agent import CaseStrategyAgent
from src.agents.cyber_fraud_agent import CyberFraudAgent
from src.agents.police_complaint_agent import PoliceComplaintAgent
from src.agents.labour_dues_agent import LabourDuesAgent
from src.agents.audit_agent import DocumentAuditAgent
from src.session_manager import get_session_manager
from src.case_service import CaseService
from src.evidence_service import EvidenceService
from src.analytics_service import AnalyticsService
from src.training_service import TrainingService
from src.translator import has_non_latin_script, detect_language, detect_requested_language
from src.ollama_translator import get_fast_translator
from src.llm_router import LLM_TIMEOUT_ERR
from src.response_formatter import format_legal_response
from src.domain_config import filter_sources_by_domain
from api.auth import router as auth_router
from config import API_CONFIG, MONGODB_CONFIG

load_dotenv(override=True)

# ── Logging setup ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("courtroom-api")

try:
    import structlog

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    log = structlog.get_logger()
    _USE_STRUCTLOG = True
except ImportError:
    log = logger
    _USE_STRUCTLOG = False

# ── MongoDB ──────────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGODB_URI", MONGODB_CONFIG["uri"])
MONGODB_DB = os.getenv("MONGODB_DB", MONGODB_CONFIG["db_name"])
db = MongoClient(MONGODB_URI)[MONGODB_DB]

# ── Lifespan ─────────────────────────────────────────────────────
_rag_system = None
_notice_agent = None
_evidence_agent = None
_rti_agent = None
_deadline_agent = None
_storage = None
_case_service = None
_evidence_service = None
_analytics_service = None
_training_service = None
_user_docs_service = None
_strategy_agent = None
_cyber_agent = None
_police_agent = None
_labour_agent = None
_audit_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_system, _notice_agent, _evidence_agent, _rti_agent, _deadline_agent, _storage
    global _case_service, _evidence_service, _analytics_service, _training_service, _user_docs_service
    global _strategy_agent, _cyber_agent, _police_agent, _labour_agent, _audit_agent
    log.info("Starting courtRoom.ai API...")
    try:
        _rag_system = FullRAGSystem()
        log.info("rag_system initialized")
    except Exception as e:
        log.warning("rag_system init failed", error=str(e))
    _notice_agent = LegalNoticeAgent()
    _evidence_agent = EvidenceChecklistAgent()
    _rti_agent = RTIApplicationAgent()
    _deadline_agent = DeadlineTrackerAgent()
    _strategy_agent = CaseStrategyAgent()
    _cyber_agent = CyberFraudAgent()
    _police_agent = PoliceComplaintAgent()
    _labour_agent = LabourDuesAgent()
    _audit_agent = DocumentAuditAgent()
    _storage = _QueryStorage()
    _case_service = CaseService()
    _evidence_service = EvidenceService(_case_service)
    _analytics_service = AnalyticsService()
    _training_service = TrainingService()
    try:
        _user_docs_service = UserDocsService()
        log.info("user_docs_service initialized")
    except Exception as e:
        log.warning("user_docs_service init failed", error=str(e))
    yield
    log.info("Shutting down courtRoom.ai API...")


app = FastAPI(title="courtRoom.ai API", version="1.0", lifespan=lifespan)
app.include_router(auth_router)

# ── CORS ─────────────────────────────────────────────────────────
cors_origins = API_CONFIG["cors_origins"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Rate Limiting ────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _USE_SLOWAPI = True
except ImportError:
    _USE_SLOWAPI = False


def limiter_decorator():
    if _USE_SLOWAPI:
        return limiter.limit(f"{API_CONFIG['rate_limit_per_minute']}/minute")
    return lambda fn: fn


# ── Request ID Middleware ────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    if _USE_STRUCTLOG:
        log.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=f"{duration:.0f}",
        )
    else:
        logger.info(
            "[%s] %s %s -> %d (%.0fms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
    return response


# ── Exception Handlers ───────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("X-Request-ID", "unknown")
    log.error("Unhandled exception", request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later.", "request_id": request_id},
    )


# ── Dependency Injection ─────────────────────────────────────────
async def get_rag() -> FullRAGSystem:
    if _rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return _rag_system


def get_storage():
    return _storage


def _user_profile(user_id: str) -> Optional[Dict]:
    """Fetch a logged-in user's profile (name/email) for personalization."""
    if not user_id or user_id == "anonymous":
        return None
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None
    if not user:
        return None
    return {
        "name": user.get("name", ""),
        "email": user.get("email", ""),
    }


def _tool_result_payload(query: str, tools: List[Dict]) -> Dict:
    """Final response driven by a tool result."""
    tool = tools[0]
    needs = tool.get("status") == "needs_input"
    msg = tool.get("message") or ("A few more details are needed." if needs else "Done.")
    return {
        "query": query,
        "status": "needs_more_info" if needs else "success",
        "response_type": "tool",
        "confidence_score": 0.0,
        "short_answer": msg,
        "full_response": msg,
        "response": msg,
        "sources": [],
        "applicable_laws": {},
        "tools": tools,
        "missing_facts": tool.get("missing_fields") or [],
        "questions": [] if not needs else tool.get("missing_fields") or [],
    }


# ── Multi-turn tool continuation (MongoDB-backed with TTL) ───────────────
_session_mgr = get_session_manager()


def _store_pending(user_id: str, name: str, params: Dict[str, Any]) -> None:
    if not user_id:
        return
    _session_mgr.store_session(user_id, name, params)


def _clear_pending(user_id: str) -> None:
    if not user_id:
        return
    _session_mgr.clear_session(user_id)


def _merge_pending(user_id: str, query: str, profile: Optional[Dict] = None) -> Optional[Dict]:
    """If a tool is waiting on details for this user, merge their reply.

    Only aborts the pending session if:
    1. The user sends smalltalk or a pure definition question ('what is...'), OR
    2. The user explicitly invokes a DIFFERENT tool (e.g. switching from legal_notice to rti_application)
       with an explicit drafting verb ('draft', 'file', 'generate') rather than supplying facts for the
       currently pending tool.
    """
    if not user_id:
        return None
    pending = _session_mgr.get_session(user_id)
    if not pending:
        return None
    if detect_smalltalk(query) or _DEFINITION_QUESTION_RE.match((query or "").strip()):
        _clear_pending(user_id)
        return None

    explicit_new = detect_explicit_tool(query)
    has_command_verb = bool(re.search(r"\b(draft|generate|prepare|create|make|file)\s+(?:a|an|the|my)?\b", query or "", re.I))
    if explicit_new is not None and explicit_new != pending["name"] and has_command_verb:
        _clear_pending(user_id)
        return None

    merged = merge_tool_answer(pending["name"], pending["params"], query, profile)
    result = execute_tool(pending["name"], merged, user_id, _storage, profile)
    if result.get("status") == "needs_input":
        _store_pending(user_id, pending["name"], result.get("params") or merged)
    else:
        _clear_pending(user_id)
    return result


# ── Models ───────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., max_length=API_CONFIG["max_query_length"])
    language: str = "en"


class NoticeRequest(BaseModel):
    sender_name: str
    sender_address: str = ""
    sender_email: str = ""
    recipient_name: str
    recipient_address: str = ""
    recipient_email: str = ""
    issue_type: str = "Legal Dispute"
    issue_description: str
    applicable_section: str = ""
    demand_amount: str = ""


class DeadlineRequest(BaseModel):
    case_type: str
    incident_date: str


class AuditRequest(BaseModel):
    text: str
    domain: str = "civil"


class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "hi"


class CaseCreateRequest(BaseModel):
    client_name: str = Field(..., max_length=200)
    case_type: Optional[str] = None
    description: str = ""
    status: str = "draft"
    metadata: Dict = {}


class CaseStatusRequest(BaseModel):
    status: str


# ── Storage class ─────────────────────────────────────────────────
class _QueryStorage:
    def __init__(self):
        self.db = MongoClient(MONGODB_URI)[MONGODB_DB]

    def store_query(self, user_id: str, query_data: Dict) -> str:
        document = {
            "user_id": user_id or "anonymous",
            "query": query_data["query"],
            "domain": query_data["domain"],
            "confidence": query_data["confidence"],
            "response": query_data["response"],
            "sources": query_data["sources"],
            "created_at": datetime.now(),
            "status": "completed",
        }
        result = self.db.queries.insert_one(document)
        return str(result.inserted_id)

    def store_pdf_notice(self, user_id: str, case_id: str, pdf_data: Dict) -> str:
        document = {
            "user_id": user_id or "anonymous",
            "case_id": case_id,
            "filename": pdf_data.get("filename", ""),
            "docx_filename": pdf_data.get("docx_filename", ""),
            "pdf_path": pdf_data.get("pdf_path", ""),
            "docx_path": pdf_data.get("docx_path", ""),
            "sender_name": pdf_data.get("sender_name", ""),
            "sender_email": pdf_data.get("sender_email", ""),
            "recipient_name": pdf_data.get("recipient_name", ""),
            "recipient_email": pdf_data.get("recipient_email", ""),
            "issue_type": pdf_data.get("issue_type", ""),
            "demand_amount": pdf_data.get("demand_amount", ""),
            "created_at": datetime.now(),
            "downloaded": False,
            "download_count": 0,
        }
        result = self.db.pdfs.insert_one(document)
        return str(result.inserted_id)

    def get_user_queries(self, user_id: str, limit: int = 50):
        queries = list(
            self.db.queries.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        )
        for q in queries:
            q["_id"] = str(q["_id"])
        return queries

    def get_user_pdfs(self, user_id: str):
        pdfs = list(self.db.pdfs.find({"user_id": user_id}).sort("created_at", -1))
        for p in pdfs:
            p["_id"] = str(p["_id"])
        return pdfs


# ── Translation helpers ─────────────────────────────────────────
TRANSLATION_SLA_SECONDS = float(os.getenv("TRANSLATION_SLA_SECONDS", "20"))


def _resolve_target_language(query: str, requested: str) -> str:
    requested = (requested or "en").strip().lower()
    if requested != "en":
        return requested
    detected = detect_language(query or "")
    if detected != "en":
        return detected
    return detect_requested_language(query or "")


def _translate_result(
    result: dict,
    lang: str,
    translator,
    deadline: Optional[float] = None,
) -> int:
    if lang in (None, "", "en"):
        return 0

    corrected_count = 0

    def _over_deadline() -> bool:
        if deadline is None:
            return False
        over = time.monotonic() > deadline
        if over:
            result["translated"] = "partial"
        return over

    fields_to_translate = {}
    short = result.get("short_answer")
    if isinstance(short, str) and short.strip():
        fields_to_translate["short_answer"] = short
    illegal = result.get("is_this_illegal")
    if isinstance(illegal, str) and illegal.strip():
        fields_to_translate["is_this_illegal"] = illegal
    full = result.get("full_response") or result.get("response")
    if isinstance(full, str) and full.strip():
        fields_to_translate["full_response"] = full

    if not fields_to_translate:
        return 0

    for key, src in fields_to_translate.items():
        result.setdefault(f"original_{key}", src)

    translated_fields = translator.batch_translate(fields_to_translate, lang, deadline=deadline)

    for key, translated in translated_fields.items():
        result[key] = translated
        if key == "full_response":
            result["response"] = translated

    if not _over_deadline():
        sources = result.get("sources")
        if isinstance(sources, list):
            result.setdefault("original_sources", sources)
            source_fields = {}
            for i, s in enumerate(sources):
                for field in ("section_title", "topic", "content_preview"):
                    val = s.get(field)
                    if isinstance(val, str) and val.strip():
                        source_fields[f"sources.{i}.{field}"] = val
            if source_fields:
                translated_source_fields = translator.batch_translate(
                    source_fields, lang, deadline=deadline
                )
                translated_sources = [dict(s) for s in sources]
                for key, text in translated_source_fields.items():
                    _, idx, field = key.split(".", 2)
                    translated_sources[int(idx)][field] = text
                result["sources"] = translated_sources

    return corrected_count


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
@limiter_decorator()
async def health_check(request: Request):
    import httpx

    health = {
        "status": "ok",
        "service": "courtRoom.ai",
        "version": "1.0",
    }

    try:
        db.command("ping")
        health["mongodb"] = "ok"
    except Exception as e:
        health["mongodb"] = f"error: {e}"
        health["status"] = "degraded"

    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{ollama_url}/api/tags")
            health["ollama"] = "ok" if r.status_code == 200 else "error"
    except Exception as e:
        health["ollama"] = f"error: {e}"
        health["status"] = "degraded"

    health["rag"] = "ok" if _rag_system is not None else "unavailable"
    if _rag_system is None:
        health["status"] = "degraded"

    return health


@app.post("/query")
@limiter_decorator()
async def process_query(
    req: QueryRequest,
    request: Request,
    rag: FullRAGSystem = Depends(get_rag),
    storage: _QueryStorage = Depends(get_storage),
    user_id: str = "anonymous",
):
    try:
        translator = get_fast_translator()
        original_query = req.query
        query_to_process = req.query
        target_lang = _resolve_target_language(req.query, req.language)

        if target_lang != "en" and has_non_latin_script(req.query):
            translated_query = translator.query_to_english(req.query, target_lang)
            if translated_query != req.query:
                query_to_process = translated_query

        if detect_smalltalk(query_to_process):
            _clear_pending(user_id)
            try:
                reply = await asyncio.to_thread(rag.llm_router.generate_chat_reply, query_to_process)
            except Exception:
                reply = None
            return chat_response(original_query, reply)

        profile = _user_profile(user_id)
        merged = await asyncio.to_thread(_merge_pending, user_id, query_to_process, profile)
        if merged:
            return _tool_result_payload(original_query, [merged])

        tools_route = await asyncio.to_thread(
            route_chat_tools,
            query_to_process,
            user_id,
            _storage,
            profile=profile,
        )
        if tools_route["tools"]:
            tool = tools_route["tools"][0]
            if tool.get("status") == "needs_input":
                _store_pending(user_id, tool.get("type") or "legal_notice", tool.get("params") or {})
            return _tool_result_payload(original_query, tools_route["tools"])

        result = rag.process_query(query_to_process)
        if result.get("status") == "failed":
            return result
        result["tools"] = tools_route["tools"]
        result["generation_engine"] = rag.llm_router.generation_engine
        if not result.get("response") or not str(result.get("response")).strip():
            result["response"] = (
                result.get("full_response")
                or result.get("short_answer")
                or "No output could be generated for this query."
            )
        result["domain"] = result.get("response_type")
        result["confidence"] = result.get("confidence_score")

        if target_lang != "en":
            deadline = time.monotonic() + TRANSLATION_SLA_SECONDS
            corrected = _translate_result(result, target_lang, translator, deadline=deadline)
            result["translation_corrected"] = corrected
            result["query_language"] = target_lang
            if "translated" not in result:
                result["translated"] = True

        query_id = storage.store_query(
            user_id or "anonymous",
            {
                "query": original_query,
                "domain": result.get("response_type"),
                "confidence": result.get("confidence_score"),
                "response": result.get("response"),
                "full_response": result,
                "sources": result.get("sources", []),
            },
        )

        result["query_id"] = query_id
        result["stored"] = True

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
@limiter_decorator()
async def stream_query(
    req: QueryRequest,
    request: Request,
    rag: FullRAGSystem = Depends(get_rag),
    storage: _QueryStorage = Depends(get_storage),
    user_id: str = "anonymous",
):
    translator = get_fast_translator()
    original_query = req.query
    target_lang = _resolve_target_language(req.query, req.language)

    def sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_stream():
        try:
            query_to_process = original_query
            yield sse("status", {"step": "language", "language": target_lang, "message": ""})
            if target_lang != "en" and has_non_latin_script(original_query):
                yield sse(
                    "status",
                    {"step": "translating", "message": f"Reading your question in {target_lang.upper()}..."},
                )
                translated_query = translator.query_to_english(original_query, target_lang)
                if translated_query != original_query:
                    query_to_process = translated_query

            yield sse("status", {"step": "analyzing", "message": "Analyzing your situation..."})
            if detect_smalltalk(query_to_process):
                _clear_pending(user_id)
                yield sse("status", {"step": "chat", "message": ""})
                parts = []
                try:
                    async for piece in rag.llm_router.stream_chat_reply(query_to_process):
                        parts.append(piece)
                        if await request.is_disconnected():
                            return
                        yield sse("token", {"text": piece})
                except Exception:
                    pass
                text = "".join(parts).strip() or None
                yield sse("final", chat_response(original_query, text))
                return

            profile = _user_profile(user_id)
            merged = await asyncio.to_thread(_merge_pending, user_id, query_to_process, profile)
            if merged:
                yield sse("final", _tool_result_payload(original_query, [merged]))
                return

            tools_route = await asyncio.to_thread(
                route_chat_tools,
                query_to_process,
                user_id,
                _storage,
                profile=profile,
            )
            tools = tools_route["tools"]
            if tools:
                if tools[0].get("status") == "needs_input":
                    _store_pending(user_id, tools[0].get("type") or "legal_notice", tools[0].get("params") or {})
                yield sse("final", _tool_result_payload(original_query, tools))
                return

            prep = rag.prepare_context(query_to_process)
            if prep.get("needs_more_info"):
                body = (
                    "I need a few more details before I can identify the exact law that applies.\n\n"
                    + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(prep.get("questions") or []))
                )
                result = {
                    "query": prep["query"],
                    "status": "needs_more_info",
                    "response_type": "needs_more_info",
                    "confidence_score": 0.0,
                    "short_answer": "I need a few more details before I can identify the exact law that applies.",
                    "full_response": body,
                    "response": body,
                    "missing_facts": prep.get("missing_facts") or [],
                    "questions": prep.get("questions") or [],
                    "sources": [],
                    "applicable_laws": {},
                    "tools": tools,
                }
                yield sse("final", result)
                return

            domain = prep["domain"]
            domain_confidence = prep["domain_confidence"]
            secondary = prep["secondary"]
            sources = prep["sources"]
            if not sources:
                body = "The retrieved provisions do not establish an applicable rule for these facts."
                for r in (prep.get("rejected") or [])[:5]:
                    body += f"\n- {r.get('act_name')} {r.get('section_number')}: {r.get('reason')}"
                result = {
                    "query": prep["query"],
                    "status": "success",
                    "response_type": domain,
                    "confidence_score": prep.get("confidence_details", {}).get("overall_confidence", 0.0),
                    "short_answer": body,
                    "full_response": body,
                    "response": body,
                    "sources": [],
                    "applicable_laws": {},
                    "tools": tools,
                }
                yield sse("final", result)
                return

            context = prep["context"]
            available = prep["available"]
            query_to_process = prep["query"]

            yield sse("status", {"step": "generating", "message": "Generating legal analysis..."})
            llm_parts = []
            async for piece in rag.llm_router.stream_generate(context, query_to_process, available):
                llm_parts.append(piece)
                if await request.is_disconnected():
                    return
                yield sse("token", {"text": piece})

            llm_response = "".join(llm_parts).strip()
            if not llm_response:
                if rag.llm_router.last_was_error:
                    yield sse(
                        "status",
                        {"step": "fallback", "message": "AI service is temporarily unavailable. Please try again in a few moments."},
                    )
                    llm_response = json.dumps(LLM_TIMEOUT_ERR)
                else:
                    raise Exception("No output generated by the language model.")

            yield sse("status", {"step": "formatting", "message": "Finalizing your answer..."})
            response_type = rag.domain_classifier.get_response_type(domain, secondary)
            result = format_legal_response(
                query=query_to_process,
                llm_response=llm_response,
                sources=sources,
                domain=domain,
                confidence=domain_confidence,
                response_type=response_type,
            )
            result["generation_engine"] = rag.llm_router.generation_engine
            result["tools"] = tools
            if not result.get("response"):
                result["response"] = (
                    result.get("full_response")
                    or result.get("short_answer")
                    or "No output could be generated for this query."
                )
            result["domain"] = result.get("response_type")
            result["confidence"] = result.get("confidence_score")
            result["confidence_details"] = prep.get("confidence_details") or {}

            if target_lang != "en":
                yield sse("status", {"step": "translating", "message": f"Translating answer to {target_lang.upper()}..."})
                deadline = time.monotonic() + TRANSLATION_SLA_SECONDS
                corrected = _translate_result(result, target_lang, translator, deadline=deadline)
                result["translation_corrected"] = corrected
                result["query_language"] = target_lang
                if "translated" not in result:
                    result["translated"] = True

            query_id = storage.store_query(
                user_id or "anonymous",
                {
                    "query": original_query,
                    "domain": result.get("response_type"),
                    "confidence": result.get("confidence_score"),
                    "response": result.get("response"),
                    "full_response": result,
                    "sources": result.get("sources", []),
                },
            )
            result["query_id"] = query_id
            result["stored"] = True

            yield sse("final", result)
        except Exception as e:
            logger.exception("stream query failed")
            yield sse("error", {"detail": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/generate-notice")
async def generate_legal_notice(req: NoticeRequest, user_id: str = "anonymous"):
    try:
        case_data = req.model_dump()
        res = _notice_agent.generate_both(case_data)
        pdf_id = _storage.store_pdf_notice(
            user_id,
            "direct_api",
            {
                "filename": res["filename"],
                "docx_filename": res["docx_filename"],
                "pdf_path": res["pdf_path"],
                "docx_path": res["docx_path"],
                "sender_name": case_data.get("sender_name", ""),
                "sender_email": case_data.get("sender_email", ""),
                "recipient_name": case_data.get("recipient_name", ""),
                "recipient_email": case_data.get("recipient_email", ""),
                "issue_type": case_data.get("issue_type", "Legal Dispute"),
                "demand_amount": case_data.get("demand_amount", ""),
            },
        )

        return {
            "status": "success",
            "pdf_id": pdf_id,
            "filename": res["filename"],
            "docx_filename": res["docx_filename"],
            "download_url": f"/download/{pdf_id}?format=pdf",
            "docx_download_url": f"/download/{pdf_id}?format=docx",
            "message": "Legal notice generated in PDF and DOCX",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}/queries")
async def get_query_history(user_id: str, storage: _QueryStorage = Depends(get_storage)):
    return storage.get_user_queries(user_id)


@app.get("/user/{user_id}/pdfs")
async def get_pdf_history(user_id: str, storage: _QueryStorage = Depends(get_storage)):
    return storage.get_user_pdfs(user_id)


@app.get("/download/{doc_id}")
async def download_document(doc_id: str, format: str = "pdf"):
    fmt = (format or "pdf").lower().strip()
    try:
        obj_id = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    doc = db.pdfs.find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    db.pdfs.update_one(
        {"_id": obj_id},
        {"$inc": {"download_count": 1}, "$set": {"downloaded": True}},
    )

    if fmt == "docx":
        path = doc.get("docx_path")
        filename = doc.get("docx_filename") or f"{doc.get('filename', 'document').replace('.pdf', '')}.docx"
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        path = doc.get("pdf_path")
        filename = doc.get("filename") or "document.pdf"
        media_type = "application/pdf"

    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File for format '{fmt}' not found on server")

    return FileResponse(
        path=path,
        filename=filename,
        media_type=media_type,
    )


@app.get("/pdf/{pdf_id}/download")
async def download_pdf_legacy(pdf_id: str):
    return await download_document(doc_id=pdf_id, format="pdf")


@app.get("/evidence/{domain}")
async def get_evidence_checklist(domain: str):
    try:
        return _evidence_agent.get_checklist(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document-audit")
async def audit_document_endpoint(req: AuditRequest):
    try:
        agent = _audit_agent or DocumentAuditAgent()
        return agent.audit(req.text, req.domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
async def translate_endpoint(req: TranslateRequest):
    try:
        translator = get_fast_translator()
        translated = translator.translate(req.text, req.source_lang, req.target_lang)
        return {
            "status": "success",
            "source_lang": req.source_lang,
            "target_lang": req.target_lang,
            "original": req.text,
            "translated": translated,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rti-application")
async def generate_rti_app(data: Dict, user_id: str = "anonymous"):
    try:
        res = _rti_agent.generate_both(data)
        pdf_id = _storage.store_pdf_notice(
            user_id,
            "direct_api",
            {
                "filename": res["filename"],
                "docx_filename": res["docx_filename"],
                "pdf_path": res["pdf_path"],
                "docx_path": res["docx_path"],
                "sender_name": data.get("applicant_name", ""),
                "recipient_name": data.get("pio_office", ""),
                "issue_type": "RTI Application (Section 6)",
                "demand_amount": "Rs 10 application fee",
            },
        )
        return {
            "status": "success",
            "pdf_id": pdf_id,
            "filename": res["filename"],
            "docx_filename": res["docx_filename"],
            "download_url": f"/download/{pdf_id}?format=pdf",
            "docx_download_url": f"/download/{pdf_id}?format=docx",
            "application": res.get("application_text", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cyber-fraud")
async def report_cyber_fraud(data: Dict, user_id: str = "anonymous"):
    try:
        res = _cyber_agent.generate_filing(data)
        pdf_id = _storage.store_pdf_notice(
            user_id,
            "direct_api",
            {
                "filename": res["filename"],
                "docx_filename": res["docx_filename"],
                "pdf_path": res["pdf_path"],
                "docx_path": res["docx_path"],
                "sender_name": data.get("victim_name", ""),
                "recipient_name": data.get("victim_bank_name", "Bank"),
                "issue_type": "Cyber Fraud & Account Freeze",
                "demand_amount": data.get("amount_lost", ""),
            },
        )
        return {
            "status": "success",
            "pdf_id": pdf_id,
            "filename": res["filename"],
            "docx_filename": res["docx_filename"],
            "download_url": f"/download/{pdf_id}?format=pdf",
            "docx_download_url": f"/download/{pdf_id}?format=docx",
            "letter_text": res.get("letter_text", ""),
            "portal_instructions": res.get("portal_instructions", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/police-complaint")
async def file_police_complaint(data: Dict, user_id: str = "anonymous"):
    try:
        res = _police_agent.generate_complaint(data)
        pdf_id = _storage.store_pdf_notice(
            user_id,
            "direct_api",
            {
                "filename": res["filename"],
                "docx_filename": res["docx_filename"],
                "pdf_path": res["pdf_path"],
                "docx_path": res["docx_path"],
                "sender_name": data.get("complainant_name", ""),
                "recipient_name": data.get("police_station_name", "Police Station"),
                "issue_type": "Written Police Complaint (BNSS 2023)",
                "demand_amount": "",
            },
        )
        return {
            "status": "success",
            "pdf_id": pdf_id,
            "filename": res["filename"],
            "docx_filename": res["docx_filename"],
            "download_url": f"/download/{pdf_id}?format=pdf",
            "docx_download_url": f"/download/{pdf_id}?format=docx",
            "complaint_text": res.get("complaint_text", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/labour-dues")
async def claim_labour_dues(data: Dict, user_id: str = "anonymous"):
    try:
        res = _labour_agent.generate_demand(data)
        pdf_id = _storage.store_pdf_notice(
            user_id,
            "direct_api",
            {
                "filename": res["filename"],
                "docx_filename": res["docx_filename"],
                "pdf_path": res["pdf_path"],
                "docx_path": res["docx_path"],
                "sender_name": data.get("employee_name", ""),
                "recipient_name": data.get("company_name", ""),
                "issue_type": "Salary & Labour Dues Claim",
                "demand_amount": data.get("total_dues_amount", ""),
            },
        )
        return {
            "status": "success",
            "pdf_id": pdf_id,
            "filename": res["filename"],
            "docx_filename": res["docx_filename"],
            "download_url": f"/download/{pdf_id}?format=pdf",
            "docx_download_url": f"/download/{pdf_id}?format=docx",
            "notice_text": res.get("notice_text", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategy")
async def build_strategy(data: Dict, user_id: str = "anonymous"):
    description = str(data.get("case_description") or "").strip()
    if not description:
        raise HTTPException(status_code=400, detail="case_description is required")
    try:
        strategy = _strategy_agent.build(
            description=description,
            domain=data.get("domain"),
            incident_date=data.get("incident_date"),
        )
        pdf_id = ""
        if strategy.get("pdf_path") and _storage:
            comp = strategy.get("compensation_estimate", {})
            pdf_id = _storage.store_pdf_notice(
                user_id,
                "direct_api",
                {
                    "filename": strategy.get("filename", ""),
                    "docx_filename": strategy.get("docx_filename", ""),
                    "pdf_path": strategy.get("pdf_path", ""),
                    "docx_path": strategy.get("docx_path", ""),
                    "sender_name": "Case Brief",
                    "recipient_name": strategy.get("domain", ""),
                    "issue_type": "Case Strategy & Legal Opinion",
                    "demand_amount": f"{comp.get('min_amount', 0)}-{comp.get('max_amount', 0)}",
                },
            )
        return {
            "status": "success",
            "pdf_id": pdf_id,
            "download_url": f"/download/{pdf_id}?format=pdf" if pdf_id else "",
            "docx_download_url": f"/download/{pdf_id}?format=docx" if pdf_id else "",
            **strategy,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deadline")
async def check_deadline(req: DeadlineRequest):
    try:
        result = _deadline_agent.calculate_deadline(req.case_type, req.incident_date)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Case Management (Phase 11) ───────────────────────────────────
@app.post("/cases")
async def create_case(req: CaseCreateRequest, user_id: str = "anonymous"):
    try:
        case = _case_service.create_case(
            user_id=user_id,
            client_name=req.client_name,
            case_type=req.case_type or "general",
            description=req.description,
            status=req.status,
            metadata=req.metadata,
        )
        return {"status": "success", "case": case}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cases")
async def list_cases(user_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50):
    return _case_service.list_cases(user_id=user_id, status=status, limit=limit)


@app.get("/cases/{case_id}")
async def get_case(case_id: str):
    case = _case_service.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@app.patch("/cases/{case_id}/status")
async def update_case_status(case_id: str, req: CaseStatusRequest):
    try:
        case = _case_service.update_case_status(case_id, req.status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@app.delete("/cases/{case_id}")
async def delete_case(case_id: str):
    if not _case_service.delete_case(case_id):
        raise HTTPException(status_code=404, detail="Case not found")
    return {"status": "success", "deleted": case_id}


# ── Evidence Upload (Phase 12) ───────────────────────────────────
@app.post("/cases/{case_id}/evidence")
async def upload_evidence(
    case_id: str,
    file: UploadFile = File(...),
    uploaded_by: str = "anonymous",
):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        evidence = _evidence_service.save_evidence(
            case_id=case_id,
            filename=file.filename or "unnamed",
            content_type=file.content_type or "application/octet-stream",
            data=data,
            uploaded_by=uploaded_by,
        )
        if not evidence:
            raise HTTPException(status_code=404, detail="Case not found")
        return {"status": "success", "evidence": evidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cases/{case_id}/evidence")
async def list_evidence(case_id: str):
    return _evidence_service.list_evidence(case_id)


@app.get("/evidence/{file_id}/download")
async def download_evidence(file_id: str):
    result = _evidence_service.get_evidence(file_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evidence file not found")
    meta, data = result
    filename = meta.get("filename", "evidence.bin")
    return Response(
        content=data,
        media_type=meta.get("content_type") or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/evidence/{file_id}")
async def delete_evidence(file_id: str):
    if not _evidence_service.delete_evidence(file_id):
        raise HTTPException(status_code=404, detail="Evidence file not found")
    return {"status": "success", "deleted": file_id}


# ── User Document Upload (Google embeddings) ─────────────────────────
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))


def _get_user_docs_service() -> UserDocsService:
    if _user_docs_service is None:
        raise HTTPException(status_code=503, detail="Document service not initialized")
    return _user_docs_service


@app.post("/chat/upload")
async def chat_upload(file: UploadFile = File(...), user_id: str = "anonymous"):
    service = _get_user_docs_service()
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 25MB)")
    try:
        result = service.ingest_pdf(user_id, data, file.filename or "upload.pdf")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("chat upload failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "status": "success",
        "message": f"Uploaded {result['filename']} ({result['chunk_count']} chunks)",
        **result,
        "model": getattr(service.provider, "model", service.provider.name),
    }


@app.post("/chat/search-docs")
async def chat_search_docs(query: str, user_id: str = "anonymous", top_k: int = 5):
    service = _get_user_docs_service()
    if not query or len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters")
    results = service.search(user_id, query.strip(), top_k=min(top_k, 10))
    return {"query": query.strip(), "results_count": len(results), "results": results}


@app.get("/user/{user_id}/documents")
async def list_user_documents(user_id: str):
    service = _get_user_docs_service()
    return service.list_docs(user_id)


# ── Analytics (Phase 13) ───────────────────────────────────────────
@app.get("/analytics/overview")
async def analytics_overview(days: int = 30):
    return _analytics_service.overview(days=days)


# ── Training Data Export (Phase 19) ───────────────────────────────
@app.post("/training/export-dataset")
async def export_training_dataset(
    output_path: str = "training_data.jsonl",
    min_confidence: float = 0.5,
):
    return _training_service.export_dataset(
        output_path=output_path,
        min_confidence=min_confidence,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True,
    )
