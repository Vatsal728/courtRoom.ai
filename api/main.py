import sys
import os
import uuid
import time
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime

from src.full_rag import FullRAGSystem
from src.agents.notice_agent import LegalNoticeAgent
from src.agents.evidence_agent import EvidenceChecklistAgent
from src.agents.rti_agent import RTIApplicationAgent
from src.agents.deadline_agent import DeadlineTrackerAgent
from src.case_service import CaseService
from src.evidence_service import EvidenceService
from src.analytics_service import AnalyticsService
from src.training_service import TrainingService
from src.translator import get_translator, has_non_latin_script, detect_language
from src.response_formatter import format_legal_response
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
            structlog.processors.JSONRenderer()
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_system, _notice_agent, _evidence_agent, _rti_agent, _deadline_agent, _storage
    global _case_service, _evidence_service, _analytics_service, _training_service
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
    _storage = _QueryStorage()
    _case_service = CaseService()
    _evidence_service = EvidenceService(_case_service)
    _analytics_service = AnalyticsService()
    _training_service = TrainingService()
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
            request_id, request.method, request.url.path, response.status_code, duration
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


# ── Models ───────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., max_length=API_CONFIG["max_query_length"])
    language: str = "en"


class NoticeRequest(BaseModel):
    sender_name: str
    sender_address: str
    recipient_name: str
    recipient_address: str
    issue_type: str
    issue_description: str
    applicable_section: str
    demand_amount: str


class DeadlineRequest(BaseModel):
    case_type: str
    incident_date: str


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
            "user_id": user_id,
            "query": query_data["query"],
            "domain": query_data["domain"],
            "confidence": query_data["confidence"],
            "response": query_data["response"],
            "sources": query_data["sources"],
            "created_at": datetime.now(),
            "status": "completed"
        }
        result = self.db.queries.insert_one(document)
        return str(result.inserted_id)

    def store_pdf_notice(self, user_id: str, case_id: str, pdf_data: Dict) -> str:
        document = {
            "user_id": user_id,
            "case_id": case_id,
            "filename": pdf_data["filename"],
            "pdf_path": pdf_data["pdf_path"],
            "sender_name": pdf_data["sender_name"],
            "recipient_name": pdf_data["recipient_name"],
            "issue_type": pdf_data["issue_type"],
            "demand_amount": pdf_data["demand_amount"],
            "created_at": datetime.now(),
            "downloaded": False,
            "download_count": 0
        }
        result = self.db.pdfs.insert_one(document)
        return str(result.inserted_id)

    def get_user_queries(self, user_id: str, limit: int = 50):
        queries = list(self.db.queries.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit))
        for q in queries:
            q["_id"] = str(q["_id"])
        return queries

    def get_user_pdfs(self, user_id: str):
        pdfs = list(self.db.pdfs.find(
            {"user_id": user_id}
        ).sort("created_at", -1))
        for p in pdfs:
            p["_id"] = str(p["_id"])
        return pdfs


# ── Translation helpers ─────────────────────────────────────────
def _resolve_target_language(query: str, requested: str) -> str:
    """Effective answer language for a query.

    The dropdown selection wins; when it is English but the query was typed
    in an Indian script (Gujarati/Hindi/etc.), auto-detect the language and
    answer in it, so no manual selection is needed.
    """
    requested = (requested or "en").strip().lower()
    if requested != "en":
        return requested
    detected = detect_language(query or "")
    return detected if detected != "en" else "en"


def _translate_result(result: dict, lang: str, translator) -> None:
    """Translate the structured answer + source previews into `lang`, in place.

    Full statute text in sources stays English to keep translation fast.
    Original English values are preserved under original_* keys.
    """
    if lang in (None, "", "en"):
        return

    def _keep_original(key: str, val):
        result.setdefault(f"original_{key}", val)

    short = result.get("short_answer")
    if isinstance(short, str) and short.strip():
        _keep_original("short_answer", short)
        result["short_answer"] = translator.answer_in_language(short, lang)

    full = result.get("full_response") or result.get("response")
    if isinstance(full, str) and full.strip():
        _keep_original("full_response", full)
        translated_full = translator.answer_in_language(full, lang)
        result["full_response"] = translated_full
        result["response"] = translated_full

    sources = result.get("sources")
    if isinstance(sources, list):
        _keep_original("sources", sources)
        translated_sources = []
        for s in sources:
            ts = dict(s)
            for field in ("section_title", "topic"):
                val = ts.get(field)
                if isinstance(val, str) and val.strip():
                    ts[field] = translator.answer_in_language(val, lang)
            preview = ts.get("content_preview")
            if isinstance(preview, str) and preview.strip():
                ts["content_preview"] = translator.answer_in_language(preview, lang)
            translated_sources.append(ts)
        result["sources"] = translated_sources


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/live-location/data")
async def live_location_stub(request: Request):
    return {"enabled": False, "message": "Live location tracking is not available in this version."}

@app.get("/health")
@limiter_decorator()
async def health_check(request: Request):
    import httpx

    health = {
        "status": "ok",
        "service": "courtRoom.ai",
        "version": "1.0"
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
async def process_query(req: QueryRequest, request: Request, rag: FullRAGSystem = Depends(get_rag),
                        storage: _QueryStorage = Depends(get_storage)):
    try:
        translator = get_translator()
        original_query = req.query
        query_to_process = req.query
        target_lang = _resolve_target_language(req.query, req.language)

        if target_lang != "en" and has_non_latin_script(req.query):
            translated_query = translator.query_to_english(req.query, target_lang)
            if translated_query != req.query:
                query_to_process = translated_query

        result = rag.process_query(query_to_process)
        if result.get("status") == "failed":
            return result
        if not result.get("response") or not str(result.get("response")).strip():
            result["response"] = result.get("full_response") or result.get("short_answer") or "No output could be generated for this query."
        result["domain"] = result.get("response_type")
        result["confidence"] = result.get("confidence_score")

        if target_lang != "en":
            _translate_result(result, target_lang, translator)
            result["query_language"] = target_lang
            result["translated"] = True

        query_id = storage.store_query("anonymous", {
            "query": original_query,
            "domain": result.get("response_type"),
            "confidence": result.get("confidence_score"),
            "response": result.get("response"),
            "full_response": result,
            "sources": result.get("sources", [])
        })

        result["query_id"] = query_id
        result["stored"] = True

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
@limiter_decorator()
async def stream_query(req: QueryRequest, request: Request, rag: FullRAGSystem = Depends(get_rag),
                       storage: _QueryStorage = Depends(get_storage)):
    translator = get_translator()
    original_query = req.query
    target_lang = _resolve_target_language(req.query, req.language)

    def sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_stream():
        try:
            query_to_process = original_query
            if target_lang != "en" and has_non_latin_script(original_query):
                yield sse("status", {"step": "translating", "message": f"Reading your question in {target_lang.upper()}..."})
                translated_query = translator.query_to_english(original_query, target_lang)
                if translated_query != original_query:
                    query_to_process = translated_query

            yield sse("status", {"step": "retrieving", "message": "Searching applicable laws..."})
            domain, domain_confidence, _ = rag.domain_classifier.classify(query_to_process)
            sources = rag.improved_rag.retrieve_with_metadata(query_to_process, top_k=5)
            context = "\n\n".join(
                f"[{s.get('act_name') or s.get('source_act')} Section "
                f"{s.get('section_number') or s.get('section')} - {s.get('section_title') or ''}]\n{s['content']}"
                for s in sources
            )
            available = "\n".join(
                f"{s.get('act_name') or s.get('source_act')} Section "
                f"{s.get('section_number') or s.get('section')}: {s.get('section_title') or ''}"
                for s in sources
            )

            yield sse("status", {"step": "generating", "message": "Generating legal analysis..."})
            llm_parts = []
            async for piece in rag.llm_router.stream_generate(context, query_to_process, available):
                llm_parts.append(piece)
                if await request.is_disconnected():
                    return
                yield sse("token", {"text": piece})

            llm_response = "".join(llm_parts).strip()
            if not llm_response:
                raise Exception("No output generated by the language model.")

            yield sse("status", {"step": "formatting", "message": "Finalizing your answer..."})
            result = format_legal_response(
                query=query_to_process,
                llm_response=llm_response,
                sources=sources,
                domain=domain,
                confidence=domain_confidence
            )
            if not result.get("response"):
                result["response"] = result.get("full_response") or result.get("short_answer") or "No output could be generated for this query."
            result["domain"] = result.get("response_type")
            result["confidence"] = result.get("confidence_score")

            if target_lang != "en":
                yield sse("status", {"step": "translating", "message": f"Translating answer to {target_lang.upper()}..."})
                _translate_result(result, target_lang, translator)
                result["query_language"] = target_lang
                result["translated"] = True

            query_id = storage.store_query("anonymous", {
                "query": original_query,
                "domain": result.get("response_type"),
                "confidence": result.get("confidence_score"),
                "response": result.get("response"),
                "full_response": result,
                "sources": result.get("sources", [])
            })
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
        filename = f"Legal_Notice_{case_data['sender_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        case_data['filename'] = filename

        pdf_path = _notice_agent.generate_notice(case_data, output_path=f"output/{filename}")

        pdf_id = _storage.store_pdf_notice(user_id, "case_001", {
            "filename": filename,
            "pdf_path": pdf_path,
            "sender_name": case_data["sender_name"],
            "recipient_name": case_data["recipient_name"],
            "issue_type": case_data["issue_type"],
            "demand_amount": case_data["demand_amount"]
        })

        return {
            "status": "success",
            "pdf_id": pdf_id,
            "pdf_path": pdf_path,
            "filename": filename,
            "message": "Legal notice generated and saved"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}/queries")
async def get_query_history(user_id: str, storage: _QueryStorage = Depends(get_storage)):
    return storage.get_user_queries(user_id)


@app.get("/user/{user_id}/pdfs")
async def get_pdf_history(user_id: str, storage: _QueryStorage = Depends(get_storage)):
    return storage.get_user_pdfs(user_id)


@app.get("/pdf/{pdf_id}/download")
async def download_pdf(pdf_id: str):
    pdf_doc = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
    if not pdf_doc:
        raise HTTPException(status_code=404, detail="PDF not found")

    db.pdfs.update_one(
        {"_id": ObjectId(pdf_id)},
        {"$inc": {"download_count": 1}, "$set": {"downloaded": True}}
    )

    return FileResponse(
        path=pdf_doc["pdf_path"],
        filename=pdf_doc["filename"],
        media_type="application/pdf"
    )


@app.get("/evidence/{domain}")
async def get_evidence_checklist(domain: str):
    try:
        return _evidence_agent.get_checklist(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rti-application")
async def generate_rti_app(data: Dict):
    try:
        rti_app = _rti_agent.generate_rti_application(data)
        return {"status": "success", "application": rti_app}
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
            metadata=req.metadata
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
async def upload_evidence(case_id: str, file: UploadFile = File(...),
                          uploaded_by: str = "anonymous"):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        evidence = _evidence_service.save_evidence(
            case_id=case_id,
            filename=file.filename or "unnamed",
            content_type=file.content_type or "application/octet-stream",
            data=data,
            uploaded_by=uploaded_by
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
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.delete("/evidence/{file_id}")
async def delete_evidence(file_id: str):
    if not _evidence_service.delete_evidence(file_id):
        raise HTTPException(status_code=404, detail="Evidence file not found")
    return {"status": "success", "deleted": file_id}


# ── Analytics (Phase 13) ───────────────────────────────────────────
@app.get("/analytics/overview")
async def analytics_overview(days: int = 30):
    return _analytics_service.overview(days=days)


# ── Training Data Export (Phase 19) ───────────────────────────────
@app.post("/training/export-dataset")
async def export_training_dataset(
    output_path: str = "training_data.jsonl",
    min_confidence: float = 0.5
):
    return _training_service.export_dataset(
        output_path=output_path,
        min_confidence=min_confidence
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True
    )
