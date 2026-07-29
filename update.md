# courtRoom.ai — Full Improvement Plan

> Generated: July 2026
> Covers: Backend (Python/FastAPI) + Frontend (React/TypeScript)

---

## Contents

1. [Architecture Overview (Post-Change)](#1-architecture-overview-post-change)
2. [Files to Delete](#2-files-to-delete)
3. [Files to Create](#3-files-to-create)
4. [Phase 1: Quick Wins (1-2 days)](#4-phase-1-quick-wins-1-2-days)
5. [Phase 2: RAG Quality (3-5 days)](#5-phase-2-rag-quality-3-5-days)
6. [Phase 3: Response Quality (3-4 days)](#6-phase-3-response-quality-3-4-days)
7. [Phase 4: Architecture & Infrastructure (5-7 days)](#7-phase-4-architecture--infrastructure-5-7-days)
8. [vLLM on Google Colab](#8-vllm-on-google-colab)
9. [Testing Plan](#9-testing-plan)
10. [Effort Summary](#10-effort-summary)

---

## 1. Architecture Overview (Post-Change)

```
User Query
    │
    ▼
[NLP Pipeline] → [Domain Classifier (keywords + NaiveBayes)]
    │
    ▼
[RAG Pipeline — Hybrid Retrieval]
  ┌──────────────────────────────┐
  │  ChromaDB (dense, MMR)       │
  │  BM25 (sparse, dynamic wgt)  │
  │  BGE Reranker V2             │  ← NEW: rerank top 20 → top 5
  │  HyDE                        │  ← NEW: hypothetical doc embedding
  └──────────────────────────────┘
    │
    ▼
[LLMRouter]
  ├─ Colab vLLM (Qwen 7B / Mistral 7B)  ── primary GPU (via ngrok)
  ├─ Local Ollama (same model)           ── fallback
  │  Both use format: "json"
  └─ Returns structured JSON directly
    │
    ▼
[ResponseFormatter] ── validates JSON schema, fills defaults
    │
    ▼
Structured JSON Response → Frontend renders from fields
```

### What's Removed
- **Gemini** (`llm_router.py`): No more Google API dependency
- **FOL Inference Engine** (`inference_engine.py` + `knowledge_base.py`): Dead code, LLM handles legal reasoning better
- **Dead code** in `full_rag.py` (unused `CaseTypeClassifier`, `NLPPipeline`)
- **Regex post-processing** in `response_formatter.py`: LLM returns structured JSON directly
- **IPC→BNS mapping post-processing**: Handled in the LLM prompt now

### What Stays
- `nlp_pipeline.py` — will be upgraded (language detection, spaCy NER)
- `domain_classifier.py` — will be upgraded (keyword + NaiveBayes ensemble)
- `rag_pipeline.py` — major quality upgrades
- `llm_router.py` — rewritten, no Gemini
- `response_formatter.py` — rewritten, no regex
- `full_rag.py` — simplified
- `api/main.py` — async + DI + logging
- All agents (`notice_agent.py`, `evidence_agent.py`, `rti_agent.py`, `deadline_agent.py`) — untouched

---

## 2. Files to Delete

| File | Reason |
|---|---|
| `src/inference_engine.py` | Dead code, not called anywhere |
| `src/knowledge_base.py` | Rules only used by inference_engine |

---

## 3. Files to Create

| File | Purpose |
|---|---|
| `colab_vllm_server.ipynb` | Colab notebook to run vLLM with ngrok tunnel |
| `tests/test_domain_classifier.py` | Unit tests — all domains, edge cases, negation |
| `tests/test_rag_pipeline.py` | Unit tests — mocked chroma/bm25, reranker |
| `tests/test_response_formatter.py` | Unit tests — JSON schema validation |
| `tests/test_llm_router.py` | Unit tests — fallback chain, retry logic |
| `tests/test_api.py` | Integration tests — FastAPI TestClient |

---

## 4. Phase 1: Quick Wins (1-2 days)

### 4.1 Simplify LLMRouter — Remove Gemini
**File:** `llm_router.py`
**Changes:**
- Remove all Gemini code (`get_gemini_response`, `google.generativeai` import)
- Rename `get_ollama_response` → `get_llm_response`
- New fallback chain: `try Colab vLLM → fallback to Local Ollama → return error`
- Extract hardcoded prompt to class constant `SYSTEM_PROMPT`
- Add `"format": "json"` to Ollama request body
- Add JSON schema description to the prompt so LLM knows the expected structure
- Make `COLAB_VLLM_URL`, `COLAB_VLLM_MODEL` env-configurable

### 4.2 Update `.env`
**File:** `.env`
**Changes:**
- Remove `GEMINI_API_KEY`
- Add:
  ```
  COLAB_VLLM_URL=
  COLAB_VLLM_MODEL=qwen2.5:7b
  ```

### 4.3 Update `requirements.txt`
**File:** `requirements.txt`
**Changes:**
- Remove `google-generativeai`, `langchain-google-genai`

### 4.4 Fix DomainClassifier Keyword Matching
**File:** `domain_classifier.py`
**Changes:**
- Replace all `if keyword in query_lower` with `re.search(rf'\b{re.escape(kw)}\b', query_lower)`
- Prevents false positives like "rent" matching "current" or "parent"

### 4.5 Clean Up FullRAGSystem
**File:** `full_rag.py`
**Changes:**
- Remove `self.classifier` (CaseTypeClassifier instance) — unused
- Remove `self.nlp` (NLPPipeline instance) — unused in `process_query`
- Remove corresponding imports
- Simplify `__init__` to only: `domain_classifier`, `response_formatter`, `improved_rag`, `llm_router`

### 4.6 Persist BM25 with joblib
**File:** `rag_pipeline.py`
**Changes:**
- In `build_knowledge_base()`: after creating BM25, serialize docs with `joblib.dump(all_chunks, "data/chunks/bm25_docs.pkl")`
- In `__init__()`: load with `joblib.load("data/chunks/bm25_docs.pkl")` instead of JSON
- Remove `all_chunks_langchain.json` — no longer needed
- Add `import joblib`
- Add `joblib` to `requirements.txt` (if not already present)

### 4.7 Remove Redundant IPC→BNS Mapping
**File:** `response_formatter.py`
**Changes:**
- Remove the `ipc_bns_map` regex block (lines 26-36) in `clean_markdown()`
- The LLM prompt already handles this mapping

### 4.8 Delete Dead Files
- Delete `src/inference_engine.py`
- Delete `src/knowledge_base.py`

### 4.9 Update API Main
**File:** `api/main.py`
**Changes:**
- Remove `InferenceEngine`, `NLPPipeline`, `CaseTypeClassifier` imports
- Remove `inference = InferenceEngine()`, `nlp = NLPPipeline()`, `classifier = CaseTypeClassifier()` instances
- Add MongoDB and Ollama health checks to `/health` endpoint
- Return `{"db": "ok", "ollama": "ok/error"}` alongside existing fields

---

## 5. Phase 2: RAG Quality (3-5 days)

### 5.1 Add BGE Reranker V2
**File:** `rag_pipeline.py`
**Changes:**
- After ensemble retrieves top 20 docs, pass through `BAAI/bge-reranker-v2-m3`
- Use `sentence_transformers.CrossEncoder` with ONNX runtime for speed
- Keep top 5 reranked results
- Add `optimum[onnxruntime]` and `sentence-transformers` to `requirements.txt`
- Add `RERANKER_MODEL`, `RERANKER_TOP_K` to `config.py`

**Implementation sketch:**
```python
from sentence_transformers import CrossEncoder

class RAGPipeline:
    def __init__(self):
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cpu"  # or "cuda" if GPU available
        )
    
    def retrieve_with_metadata(self, query, top_k=5):
        initial_results = self._ensemble_retrieve(query, top_k=20)
        pairs = [(query, r["content"]) for r in initial_results]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(initial_results, scores),
            key=lambda x: x[1], reverse=True
        )
        return [r for r, s in ranked[:top_k]]
```

### 5.2 Add MMR to ChromaDB
**File:** `rag_pipeline.py`
**Changes:**
- Set `search_type="mmr"` on ChromaDB retriever
- `search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}`
- Reduces redundancy in retrieved chunks

### 5.3 Dynamic Ensemble Weights
**File:** `rag_pipeline.py`
**Changes:**
- If query matches `\b(?:section|article|rule|IPC|BNS|Section)\s*\d+` → use weights `[0.3, 0.7]` (favor BM25)
- Else → use default `[0.6, 0.4]` (favor vector)
- Makes retrieval adaptive to query type

### 5.4 Add HyDE (Hypothetical Document Embeddings)
**File:** `rag_pipeline.py`
**Changes:**
- Before retrieval, send query to Ollama with prompt: `"Generate a hypothetical legal scenario describing: {query}"`
- Embed the generated text instead of the raw query
- Improves semantic search for short/narrative queries
- Make HyDE prompt configurable in `config.py`

### 5.5 Centralize Configuration
**File:** `config.py`
**Changes:**
- Add all hardcoded constants from `rag_pipeline.py`:
  - `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`
  - `ENSEMBLE_TOP_K=20`, `FINAL_TOP_K=5`
  - `ENSEMBLE_WEIGHTS_DEFAULT=[0.6, 0.4]`
  - `ENSEMBLE_WEIGHTS_BM25_FAVORED=[0.3, 0.7]`
  - `RERANKER_MODEL="BAAI/bge-reranker-v2-m3"`
  - `HYDE_ENABLED=True`
  - `MMR_LAMBDA_MULT=0.7`

---

## 6. Phase 3: Response Quality (3-4 days)

### 6.1 LLM JSON Output Schema
**File:** `llm_router.py`
**Changes:**
- Define a Pydantic-style schema description in the prompt for LLM to follow
- Ollama's `"format": "json"` forces valid JSON output
- Example schema the LLM must return:
```json
{
  "short_answer": "...",
  "is_this_illegal": true,
  "illegal_explanation": "...",
  "criminal_route": {
    "applicable_sections": ["BNS 2023 Section ..."],
    "penalties": ["..."],
    "procedure": ["..."]
  },
  "civil_route": {
    "remedies": ["..."],
    "compensation": {"min": 0, "max": 50000, "currency": "INR", "type": "compensatory"},
    "procedure": ["..."]
  },
  "compensation_claims": ["..."],
  "evidence_needed": ["..."],
  "practical_steps": ["..."]
}
```

### 6.2 Rewrite ResponseFormatter
**File:** `response_formatter.py`
**Changes:**
- Remove all regex parsing logic (~200 lines)
- New logic:
  1. Parse LLM JSON response with `json.loads()`
  2. Validate against expected keys/schema
  3. Fill missing fields with sensible defaults
  4. Return structured dict directly
- Target: ~80 lines, no regex

### 6.3 Update FullRAGSystem.process_query
**File:** `full_rag.py`
**Changes:**
- Mark `format="json"` when calling `llm_router.generate_response()`
- Pass the JSON response through new `ResponseFormatter`
- Remove the compatibility key-overwriting kludge (lines 87-91)

### 6.4 DomainClassifier + NaiveBayes Ensemble
**File:** `domain_classifier.py`
**Changes:**
- Import `CaseTypeClassifier` inside class
- In `classify()`: get keyword scores + NaiveBayes probabilities
- `final_score = 0.6 * normalized_keyword_score + 0.4 * nb_probability`
- Return combined confidence

### 6.5 DomainClassifier — Negation Detection
**File:** `domain_classifier.py`
**Changes:**
- Add regex pattern: `\b(not|no|never|didn't|wasn't|won't)\s+\w*(?:criminal|crime|illegal|theft|assault|fraud|harassment)\b`
- When matched, subtract 3 from that domain's keyword score

### 6.6 DomainClassifier — Multilingual Support
**File:** `domain_classifier.py`
**Changes:**
- Add Hindi keyword lists for each domain (using Devanagari script)
- Add Gujarati keyword lists for each domain
- Detect query language with `langdetect` at the start of `classify()`
- Use the appropriate keyword list based on detected language

---

## 7. Phase 4: Architecture & Infrastructure (5-7 days)

### 7.1 Async API Endpoints
**File:** `api/main.py`
**Changes:**
- Convert all endpoints to `async def`
- Use `httpx.AsyncClient()` for Ollama/vLLM API calls
- Use `motor` instead of `pymongo` for async MongoDB access
- Add `httpx` and `motor` to `requirements.txt`

### 7.2 Dependency Injection
**File:** `api/main.py`
**Changes:**
- Replace global `rag_system`, `storage`, etc. with FastAPI `Depends()`
- Use `@asynccontextmanager` lifespan to initialize/shutdown resources
- Enables easier testing (mock dependencies per test)

### 7.3 Structured Logging
**File:** `api/main.py`
**Changes:**
- Add `structlog` for structured JSON logging
- Add request ID middleware (`uuid4` per request)
- Log: request path, duration, status code, user_id, query length
- Log errors with full traceback

### 7.4 Rate Limiting
**File:** `api/main.py`
**Changes:**
- Add `slowapi` middleware
- `/query`: 30 requests/min per IP
- `/generate-notice`, `/rti-application`: 10 requests/min per IP
- Static files / health: unlimited

### 7.5 Security Hardening
**File:** `api/main.py`
**Changes:**
- Restrict CORS `allow_origins` to frontend URL (not `*`)
- Add custom exception handlers that return user-safe messages (no stack traces)
- Add request size limits (e.g., max query length 2000 chars)

### 7.6 Query Caching
**File:** `full_rag.py`
**Changes:**
- Add `cachetools.TTLCache(maxsize=100, ttl=3600)`
- Cache key: `(query.lower().strip(), domain)`
- Check cache before calling LLM
- Invalidate on write operations

### 7.7 Structured Exceptions
**File:** `full_rag.py`
**Changes:**
- Define exception classes: `RAGError`, `LLMError`, `ClassificationError`, `RetrievalError`
- Replace all bare `except Exception` with specific types
- Each exception carries `user_message` (safe to show) and `tech_message` (logged)

### 7.8 LLM Router — Retry Logic
**File:** `llm_router.py`
**Changes:**
- Add retry wrapper: 2 attempts per provider
- Exponential backoff: 1s, then 3s
- Only retry on connection/timeout errors (not 4xx)

### 7.9 LLM Router — Graceful Fallback
**File:** `llm_router.py`
**Changes:**
- Third fallback: return a static error response dict
- `{"error": True, "message": "AI service unavailable. Please try again later.", "short_answer": "..."}`

### 7.10 Upgrade NLP Pipeline
**File:** `nlp_pipeline.py`
**Changes:**
- Replace NLTK `ne_chunk` with spaCy `en_core_web_sm`
- Add `langdetect` for language detection
- Add English-only and Hindi-only processing branches
- Return detected language in result dict

---

## 8. vLLM on Google Colab

### 8.1 Architecture

```
Google Colab (free T4)                Your Machine
┌─────────────────────────┐          ┌──────────────────────┐
│                         │   HTTP   │  LLMRouter            │
│  vLLM Server            │◄─────────│                       │
│  Model: Qwen2.5-7B      │  ngrok   │  1. Try Colab vLLM    │
│  or Mistral-7B          │  tunnel  │  2. Fail → Local      │
│  format: json           │          │     Ollama            │
│  Temperature: 0.1       │          │  3. Fail → error      │
└─────────────────────────┘          └──────────────────────┘
```

### 8.2 Colab Notebook (`colab_vllm_server.ipynb`)

```python
# Cell 1: Install dependencies
!pip install vllm pyngrok

# Cell 2: Setup ngrok
from google.colab import userdata
from pyngrok import ngrok
ngrok.set_auth_token(userdata.get('NGROK_TOKEN'))

# Cell 3: Start vLLM OpenAI-compatible server
import subprocess, threading, time, requests

def start_vllm():
    !python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dtype auto \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --port 8001

threading.Thread(target=start_vllm, daemon=True).start()

# Wait for model to load
print("Loading model... (this takes 2-3 minutes)")
time.sleep(120)

# Verify API is running
try:
    requests.get("http://localhost:8001/v1/models")
    print("✅ vLLM API is ready")
except:
    print("⚠️  vLLM may still be loading...")

# Cell 4: Expose via ngrok
public_url = ngrok.connect(8001, "http").public_url
print(f"\n{'='*50}")
print(f"✅ vLLM API URL: {public_url}/v1")
print(f"{'='*50}")
print(f"\n📋 Set this in your .env file:")
print(f"   COLAB_VLLM_URL={public_url}")
print(f"   COLAB_VLLM_MODEL=qwen2.5:7b")

# Cell 5: Keep-alive loop (prevents idle disconnect)
while True:
    try:
        resp = requests.get(f"{public_url}/v1/models", timeout=10)
        print(f"[{time.strftime('%H:%M:%S')}] ✅ API alive")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️  API error: {e}")
    time.sleep(60)
```

### 8.3 How It Works

1. User opens Colab → opens `colab_vllm_server.ipynb`
2. **Runtime > Run all**
3. Enters ngrok token when prompted (stored in Colab secrets)
4. Waits ~2-3 min for model to load
5. Copies the ngrok URL from the output
6. Pastes into `.env`: `COLAB_VLLM_URL=https://xxxx.ngrok.io`
7. Restarts the FastAPI server
8. All queries now route through Colab vLLM

**If Colab disconnects** (idle timeout, browser close):
- API calls to Colab URL will fail
- LLMRouter automatically falls back to local Ollama
- User re-runs Colab notebook, updates URL → back to GPU speed
- Zero downtime from the user's perspective

### 8.4 Prerequisites

- Google account (free Colab)
- ngrok account (free at ngrok.com)
- ngrok auth token (from https://dashboard.ngrok.com/get-started/your-authtoken)

---

## 9. Testing Plan

### 9.1 Test Structure
```
courtRoom.ai/tests/
├── conftest.py              # Fixtures: mock RAG, mock LLM, test client
├── test_domain_classifier.py
├── test_rag_pipeline.py
├── test_response_formatter.py
├── test_llm_router.py
├── test_api.py
```

### 9.2 Test Details

| Test File | What It Tests |
|---|---|
| `test_domain_classifier.py` | All 8 domains return correct primary domain; confidence > 0.3 for clear queries; negation handling (e.g., "not criminal" returns non-criminal); Hindi/Gujarati queries; ambiguous queries return low confidence |
| `test_rag_pipeline.py` | `retrieve_with_metadata` returns at most `top_k` results; each result has expected keys; reranker changes ordering (higher relevance first); empty query returns empty list; BM25 fallback when ChromaDB unavailable |
| `test_response_formatter.py` | Valid LLM JSON returns all fields correctly; missing fields get filled with defaults; invalid JSON returns error dict; compensation field returns structured object |
| `test_llm_router.py` | Colab vLLM called when URL set; Falls back to local Ollama when Colab fails; Falls back to error when both fail; Retry logic triggers on timeout |
| `test_api.py` | `GET /health` returns 200 + expected fields; `POST /query` returns 200 + structured response; `POST /query` with empty body returns 422; Auth endpoints work; CORS headers present |

### 9.3 Test Commands
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=api --cov-report=term-missing

# Run specific test
pytest tests/test_domain_classifier.py -v
```

---

## 10. Effort Summary

| Phase | Files Changed | Files Deleted | Files Created | Days |
|---|---|---|---|---|
| **Phase 1: Quick Wins** | 7 | 2 | 0 | 1-2 |
| **Phase 2: RAG Quality** | 3 | 0 | 0 | 3-5 |
| **Phase 3: Response Quality** | 4 | 0 | 0 | 3-4 |
| **Phase 4: Architecture** | 5 | 0 | 0 | 5-7 |
| **vLLM Colab Notebook** | 0 | 0 | 1 | 1 |
| **Tests** | 0 | 0 | 6 | 2 |
| **Total** | **19** | **2** | **7** | **~3-4 weeks** |

### Dependencies Between Phases
```
Phase 1 ──────────► Phase 2 ──────────► Phase 3 ──────────► Phase 4
(no deps)           (needs Phase 1)      (needs Phase 2)      (needs Phase 1-3)

Colab Notebook ◄──────────────────────────────────────────────────┘
(independent)                                                    (needs async for full perf)

Tests ◄────────── All Phases ──────────►
(per-phase tests added in parallel)
```

### Quick Wins (Can Be Done First)
- Phase 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8 — all independent of each other
- Phase 1.6 (delete files) — after confirming Phase 1.5, 1.8
- Phase 1.9 — last, after all other Phase 1 items
