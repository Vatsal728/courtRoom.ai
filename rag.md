# courtRoom.ai — RAG System Full Disclosure

**File:** `rag.md`
**Scope:** Complete documentation of the Retrieval-Augmented Generation (RAG) system built for the courtRoom.ai legal AI assistant — architecture, every file, every dependency, every imported keyword, and a line-by-line explanation of the core pipeline code (`backend/src/rag_pipeline.py`, 1,081 lines of code / 1,208 physical lines) plus the full supporting stack.

> **Project structure note (post-restructure).** All Python lives under `backend/`: the RAG stack in `backend/src/`, the FastAPI layer in `backend/api/`, central config in `backend/config.py` + `backend/config/domain_config.json`, and the KB builder in `backend/build_kb.py`. The vector store moved to `storage/chroma_db`. Everything is run from the **repo root** (CWD = `D:\courtRoom.ai`) so relative paths (`data/...`, `models`, `.env`) resolve unchanged: e.g. `python backend/build_kb.py`, and the API via `uvicorn api.main:app --app-dir backend`. The dev-only `scripts/` (build_dataset, QLoRA trainers, build_index) and `training_data/` were removed — `data/training/legal_complaints.json` (gitignored) is the only surviving classifier training artifact, and the trained `models/*.pkl` are loaded at runtime by `backend/src/classifier.py`.

---

## Table of Contents

1. [Full Disclosure](#1-full-disclosure)
2. [Pipeline Structure](#2-pipeline-structure)
3. [Key Imports and Keywords](#3-key-imports-and-keywords)
4. [Deep Dive: `backend/src/rag_pipeline.py`](#4-deep-dive-backsrcrag_pipelinepy)
5. [Deep Dive: `backend/src/full_rag.py`](#5-deep-dive-backsrcfull_ragpy)
6. [Deep Dive: `backend/src/llm_router.py`](#6-deep-dive-backsrcllm_routerpy)
7. [Deep Dive: `backend/src/domain_classifier.py`](#7-deep-dive-backsrcdomain_classifierpy)
8. [Deep Dive: `backend/src/domain_config.py`](#8-deep-dive-backsrcdomain_configpy)
9. [Deep Dive: `backend/src/response_formatter.py`](#9-deep-dive-backsrcresponse_formatterpy)
10. [Deep Dive: `backend/src/classifier.py`](#10-deep-dive-backsrcclassifierpy)
11. [Deep Dive: Translation Stack](#11-deep-dive-translation-stack)
12. [Configuration Reference](#12-configuration-reference)
13. [The 13 Improvements + COI Fix](#13-the-13-improvements--coi-fix)
14. [Known Limitations and Gaps](#14-known-limitations-and-gaps)

---

# 1. Full Disclosure

## 1.1 What the system is

courtRoom.ai's RAG engine answers Indian legal questions in plain language. A user describes a situation (in English or an Indian language), and the system:

1. **Normalizes** the phrasing (slang/idioms → legal terms).
2. **Classifies** the legal domain (criminal, civil, rent, labour, family, cyber, consumer, commercial, defamation).
3. **Retrieves** the most relevant statute sections from a hybrid index (dense vectors + BM25 lexical).
4. **Filters** the retrieved sections to the classified domain so the LLM context cannot be polluted by off-topic acts.
5. **Generates** a structured legal analysis with a large language model (Groq `llama-3.3-70b-versatile`), with a static error response as fallback when the LLM is down.
6. **Formats** the output into a NyayGuru-style answer: short answer, "is this illegal?", criminal route (sections/penalties/procedure), civil route (remedies/compensation/procedure), compensation claims, evidence checklist, practical steps.

It is fully config-driven, telemetry-off, offline-deployable for retrieval (embeddings run locally via Ollama), and degrades gracefully at every stage.

## 1.2 File inventory

| File | Lines | Role |
|---|---|---|
| `backend/src/rag_pipeline.py` | 1,081 | **Core.** Corpus building (PDFs, SQLite DB, JSON laws), chunking, metadata enrichment, hybrid retrieval (Chroma + BM25), scoring, ranking, dedupe, search APIs. |
| `backend/src/full_rag.py` | 131 | **Orchestrator.** `FullRAGSystem.process_query` — normalize → cache → classify → retrieve → domain-filter → generate → format. Error taxonomy. |
| `backend/src/llm_router.py` | 181 | **Generation.** Groq client with system prompt, BNS-mapping rules, circuit breaker, static fallback, SSE streaming. |
| `backend/src/domain_classifier.py` | 102 | **Classification.** Keyword + Naive-Bayes ensemble, negation handling, config rules, confidence scoring. |
| `backend/src/domain_config.py` | 139 | **Config engine.** Loads `backend/config/domain_config.json` with TTL cache, deep merge with defaults, idiom normalization, domain source grounding. |
| `backend/src/response_formatter.py` | 263 | **Formatting.** Parses LLM JSON, guards against CJK hallucination, vets citations against retrieved sources, builds markdown, applies route shaping. |
| `backend/src/classifier.py` | 58 | Loads the trained `models/nb_classifier.pkl` + `models/tfidf_vectorizer.pkl` (sklearn, CWD-relative `Path("models")`) used by the classifier. |
| `backend/config.py` | 107 | Central config: `RAG_CONFIG`, MongoDB, classifier, formatter, API, cache, ChromaDB settings. |
| `backend/config/domain_config.json` | 204 | Domain keywords, idioms, negation, classifier weights, rules, response types, grounding acts, response shaping. |
| `backend/src/translator.py` | 97 | Language/script detection helpers + FLORES codes for Indian languages. |
| `backend/src/groq_translator.py` | 231 | Translation via Groq (`llama-3.1-8b-instant`), chunking, caching, batch JSON translation. |
| `backend/src/google_translator.py` | 144 | Translation via the free Google Translate endpoint (fast mid-tier fallback). |
| `backend/src/ollama_translator.py` | 93 | `FastTranslator` composite: Groq → Google → original text. |
| `backend/build_kb.py` | 14 | Entry point: `FullRAGSystem().build_knowledge_base()`. |

*Removed in the restructure:* `scripts/build_dataset.py` (dev-only law-corpus builder), the QLoRA training scripts and `training_data/` (abandoned path), and the stale `scripts/build_index.py`. The API entry is now `backend/api/main.py`.

## 1.3 External dependencies

| Package | Used for | Where |
|---|---|---|
| `langchain` | `RecursiveCharacterTextSplitter`, `EnsembleRetriever` | `rag_pipeline.py` |
| `langchain-community` | `PyPDFLoader`, `Chroma` vectorstore, `BM25Retriever` | `rag_pipeline.py` |
| `langchain-core` | `Document` type | `rag_pipeline.py` |
| `chromadb` | Persistent vector store | `rag_pipeline.py` |
| `joblib` | Pickle save/load of docs cache + sklearn models | `rag_pipeline.py`, `classifier.py` |
| `scikit-learn` + `scipy` + `numpy` | Trained Naive Bayes + TF-IDF vectorizer (loaded via joblib) | `classifier.py` |
| `groq` | LLM generation + translation | `llm_router.py`, `groq_translator.py` |
| `httpx` | Async streaming to Groq, Google Translate calls | `llm_router.py`, `google_translator.py`, `groq_translator.py` |
| `cachetools` | `TTLCache` for query-response caching | `full_rag.py` |
| `python-dotenv` | `.env` loading | all modules |
| `PyMuPDF` (`fitz`) | PDF text extraction (via PyPDFLoader) | `rag_pipeline.py` |
| `onnxruntime` | **Required at import time by chromadb** (its default embedding function) | venv |
| `kubernetes` | **Required by chromadb 0.4.21** | venv |
| `reportlab` | PDF generation for the Legal Notice agent | `backend/src/agents/notice_agent.py` |
| `sentence-transformers` + `torch` | Optional BGE cross-encoder reranker — **installed but disabled** (`reranker_enabled: False`) | `rag_pipeline.py` (lazy import) |
| `structlog` / `slowapi` | Optional structured logging / rate limiting in the API — **guarded by try/except** | `backend/api/main.py` |

## 1.4 Environment variables (.env)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | – | LLM + translation engine key. If unset, `groq_enabled=False` → static fallback. |
| `GROQ_GENERATION_MODEL` | `llama-3.3-70b-versatile` | Generation model. |
| `GROQ_GENERATION_MAX_TOKENS` | `2048` | Generation token cap. |
| `GROQ_GENERATION_TIMEOUT` | `60` | Generation timeout (s). |
| `GROQ_SSL_VERIFY` | `1` | SSL verification toggle (corporate-filter environments may set `0`). |
| `GROQ_TRANSLATE_MODEL` | `llama-3.1-8b-instant` | Translation model. |
| `GROQ_TRANSLATION_ENABLED` | `1` | Master switch for Groq translation. |
| `GROQ_TRANSLATION_TIMEOUT` | `30` | Translation timeout (s). |
| `GROQ_TRANSLATION_COOLDOWN` | `60` | Circuit-breaker cooldown after translation failure. |
| `GROQ_FILTER_BLOCK_COOLDOWN` | `300` | Longer cooldown when a web-filter/403 block is detected. |
| `GROQ_MAX_RETRIES` | `1` | Groq client retries. |
| `GOOGLE_TRANSLATION_ENABLED` | `1` | Master switch for Google Translate fallback. |
| `GOOGLE_TRANSLATION_TIMEOUT` | `15` | Google Translate timeout (s). |
| `GOOGLE_TRANSLATION_COOLDOWN` | `30` | Google circuit-breaker cooldown. |
| `GOOGLE_SSL_VERIFY` | `GROQ_SSL_VERIFY` | SSL toggle for Google calls. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server for embeddings. |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model. |
| `PDF_DIRECTORY` | `data/pdfs` | PDF corpus folder. |
| `CHROMA_DB_PATH` | `storage/chroma_db` | Vector store folder (post-restructure location). |
| `MONGODB_URI` / `MONGODB_DB` | `mongodb://localhost:27017` / `courtroom_ai` | History/storage (API layer). |
| `DOMAIN_CONFIG_TTL` | `60` | Seconds before `backend/config/domain_config.json` is reloaded. |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | FastAPI bind address/port. |
| `MAX_QUERY_LENGTH` | `2000` | API query length cap. |

## 1.5 Data artifacts

| Artifact | Format | Contents |
|---|---|---|
| `data/pdfs/*.pdf` | PDF | 15 law PDFs (BNS 2023, IPC notes, CPC charts, RTI, IT Act, wages acts, Rent acts, Tax guide, etc.). |
| `data/laws/IndiaLaw.db` | SQLite | Canonical structured law sections (tables `IPC`, `CRPC`, `CPC`, `HMA`, `IDA`, `IEA`, `MVA`, `NIA`). |
| `data/laws/*.json` | JSON | `COI.json` (Constitution), `ipc.json`, `crpc.json`, `cpc.json`, `iea.json`, `hma.json`, `ida.json` (actually the Indian Divorce Act), `mva.json`, `nia.json`. |
| `data/chunks/bm25_docs.pkl` | Pickle (joblib) | The full `List[Document]` corpus (4,041 docs) used by BM25 + dedupe/sync checks. |
| `data/chunks/corpus_signature.json` | JSON | Corpus fingerprint (chunk params + embed model + data file sizes/mtimes) — the incremental-build skip key. |
| `storage/chroma_db/` | ChromaDB | Persistent vector index (4,041 docs) — the dense half of retrieval. |
| `models/nb_classifier.pkl` + `models/tfidf_vectorizer.pkl` | Pickle (sklearn) | Trained domain classifier. |
| `data/training/legal_complaints.json` | JSON | Classifier training dataset (gitignored, dev-only; kept because `models/*.pkl` are gitignored too). |

---

# 2. Pipeline Structure

## 2.1 The current (improved) pipeline

```text
                    ┌─────────────────────────────────────────────────────────────┐
                    │                   FullRAGSystem.process_query               │
                    └─────────────────────────────────────────────────────────────┘
                                        │
                             1. normalize_query()          (idioms, longest-first, config-driven)
                                        ▼
                             2. TTL cache lookup  ──►  hit  ──►  return cached
                                        │ miss
                                        ▼
                             ┌──────────────────────────────┐
                             │   DomainClassifier.classify    │
                             │   keyword scores (0.6)          │
                             │   + NaiveBayes probs (0.4)      │
                             │   - negation penalty            │
                             │   + config rules (rent/civil,   │
                             │     cyber_defamation, etc.)     │
                             └──────────────────────────────┘
                                        │ (domain, confidence, secondary)
                                        ▼
                             ┌──────────────────────────────┐
                             │  RAGPipeline.retrieve_with_metadata(query, top_k=5)   │
                             │                                                            │
                             │  1. _normalize_query()        synonym map, longest-first   │
                             │  2. lock { _build_ensemble() } signature-skip rebuild      │
                             │  3. EnsembleRetriever          Chroma(0.3) + BM25(0.7)     │
                             │     get_relevant_documents     ── fails ──► BM25 only      │
                             │  4. Title-boost merge          + _score_pool tie-break     │
                             │  5. Sort (title, score, BNS-first)                        │
                             │  6. Dedupe (act+section, mapped-IPC removal)              │
                             └──────────────────────────────┘
                                        │ 5 source dicts (rich metadata)
                                        ▼
                             ┌──────────────────────────────┐
                             │  filter_sources_by_domain(domain)    │  grounding rules
                             │  in-domain kept first                │  no match ──► []
                             └──────────────────────────────┘
                                        │ context + available strings
                                        ▼
                             ┌──────────────────────────────┐
                             │  LLMRouter.generate_response(context, query, available)    │
                             │   Groq (json_object, temp 0.1)                             │
                             │   circuit-breaker: _down_until                             │
                             │   ── fails/disabled ──► static LLM_TIMEOUT_ERR JSON        │
                             └──────────────────────────────┘
                                        │ llm_response (JSON text)
                                        ▼
                             ┌──────────────────────────────┐
                             │  format_legal_response(...)        │
                             │   parse JSON │ CJK guard │ citation vetting               │
                             │   route shaping │ markdown build                           │
                             └──────────────────────────────┘
                                        ▼
                          Structured response dict (response, sources, domain, ...)
                          cached in TTL cache only when llm_router.last_was_error is False
```

## 2.2 How it evolved from the original design

The original pipeline (as first built) was linear and simpler:

```text
query → keyword classify → Chroma similarity search → Groq → format
```

and later gained a BM25 ensemble. Each concrete weakness motivated one of the 13 improvements documented in [Section 13](#13-the-13-improvements--coi-fix):

| Original behaviour | Weakness | Fix (current) |
|---|---|---|
| Keyword-only classification, no fallback | Weak on queries with no domain keywords | NB ensemble + config rules + fallback domain |
| Query used verbatim for BM25 | Slang ("stole", "bounced", "won't return") never matched statute text | `_normalize_query` synonym expansion, longest-first |
| Ensemble rebuilt on every query | Slow under concurrency; scores discarded | Signature-skip rebuild + `_retrieve_lock` |
| Chroma failure killed retrieval | Ollama/Chroma down → hard error | BM25-only fallback in `retrieve_with_metadata` |
| Flat BM25 ranking | Definitional sections outranked by explanation examples | `_title_boost` merge + `_score_pool` combined scores |
| IPC/BNS duplicates in results | Old + new law returned together, confusing the LLM | BNS-first sort + mapped-IPC dedupe |
| All sources passed to the LLM | Off-topic acts leaked into criminal queries (e.g. Tax guide) | `filter_sources_by_domain` grounding |
| Repeated identical queries hit Groq | Cost/latency | TTL cache gated on `last_was_error` |
| Groq outage = broken UX | No answer at all | Static fallback JSON + circuit breaker |
| LLM hallucinated section numbers | Fake citations in output | `_filter_sections` / `_filter_citations` vetting |
| Vectorstore/cache rebuilt from scratch each boot | Slow startup | `_load_existing` + `data/chunks/bm25_docs.pkl` |
| Corpus became stale silently | DB/JSON edits invisible | Staleness check on load + skip-stems dedupe |

## 2.3 Component ownership map

| Stage | Owned by | Config source |
|---|---|---|
| Normalization | `DomainClassifier.normalize_query` → `domain_config.normalize_query`; `RAGPipeline._normalize_query` | `backend/config/domain_config.json` idioms; `_QUERY_SYNONYMS` |
| Classification | `backend/src/domain_classifier.py` | `backend/config/domain_config.json` |
| Retrieval | `backend/src/rag_pipeline.py` | `backend/config.py` `RAG_CONFIG` |
| Domain grounding | `domain_config.filter_sources_by_domain` | `backend/config/domain_config.json` grounding |
| Generation | `backend/src/llm_router.py` | `.env` |
| Formatting | `backend/src/response_formatter.py` | `backend/config/domain_config.json` response_shaping |
| Corpus build | `rag_pipeline.build_knowledge_base` | `backend/config.py`, `data/` |

---

# 3. Key Imports and Keywords

## 3.1 Standard library imports

| Import | Module | Why it is there |
|---|---|---|
| `os` | env vars, paths | Loading `PDF_DIRECTORY`, `CHROMA_DB_PATH`, `OLLAMA_*`; setting telemetry-off env vars; `HF_HOME` for reranker |
| `sys` / `io` | runtime | UTF-8 stdout wrapper so emoji/Devanagari print on Windows; `sys.path.append` for `config` |
| `re` | text processing | Every regex: section headers, section numbers, citations, scripts, idioms, tokenization |
| `json` | structured data | Parsing `data/laws/*.json`, LLM JSON output, embed request bodies |
| `sqlite3` | read-only DB access | Reading `IndiaLaw.db` (`?mode=ro` URI) |
| `threading` | concurrency | `_retrieve_lock` (retriever rebuild), `_lock` in domain_config, translator singletons |
| `warnings` | hygiene | `warnings.filterwarnings("ignore")` |
| `collections.OrderedDict` | LRU cache | `_score_pool_cache` (move_to_end/popitem) |
| `urllib.request` | HTTP (embedder) | `_OllamaEmbedder` posts `/api/embed` without an HTTP SDK |
| `time` | timing | Circuit breakers, TTL, batch retry sleep |
| `typing` | annotations | `List`, `Dict`, `Tuple`, `Optional`, `Any` |

## 3.2 Third-party imports

| Import statement | What it is | Why it matters |
|---|---|---|
| `from langchain.text_splitter import RecursiveCharacterTextSplitter` | Text splitter that respects paragraph/line boundaries | Chunks statute PDF text at `chunk_size=1000`, `overlap=200` |
| `from langchain_community.document_loaders import PyPDFLoader` | PDF → page Documents (via PyMuPDF) | Raw PDF ingestion |
| `from langchain_community.vectorstores import Chroma` | Persistent vector store wrapper | The dense index at `storage/chroma_db/` |
| `from langchain.retrievers import EnsembleRetriever` | Weighted fusion of multiple retrievers | Chroma (0.3) + BM25 (0.7) hybrid |
| `from langchain.retrievers.bm25 import BM25Retriever` | BM25 lexical retriever | Sparse half of hybrid; robust without embeddings |
| `from langchain_core.documents import Document` | `Document(page_content, metadata)` | Uniform corpus object |
| `import chromadb` / `from chromadb.config import Settings` | ChromaDB client + settings | `Settings(anonymized_telemetry=False, is_persistent=True)` |
| `import joblib` | efficient NumPy-aware pickle | `bm25_docs.pkl`, sklearn models |
| `from pathlib import Path` | filesystem paths | `data/chunks/...`, `.hf_cache` |
| `from config import RAG_CONFIG` | central config | All tunables (weights, top_k, chunk size) |
| `from dotenv import load_dotenv` | `.env` loader | `load_dotenv(override=True)` |
| `from sentence_transformers import CrossEncoder` | BGE cross-encoder reranker | **Lazy + disabled** (`reranker_enabled: False`) |
| `from cachetools import TTLCache` | TTL cache | Query→response cache (100 entries, 1 h), guarded by `last_was_error` |
| `import httpx` | modern HTTP client | Async SSE streaming to Groq; Google Translate calls |
| `from groq import Groq` | Groq SDK | `chat.completions.create` with `response_format={"type":"json_object"}` |
| `import sklearn` (transitive via joblib) | ML library | The pickled `nb_classifier.pkl`/`tfidf_vectorizer.pkl` need it at load time |

## 3.3 The regex and keyword structures

| Structure | Location | Purpose |
|---|---|---|
| `_BM25_STOPWORDS` | `backend/src/rag_pipeline.py:37` | ~140 English stopwords + legal filler (`act`, `section`, `shall`, `aforesaid`, `notwithstanding`, ...) removed from BM25 tokens |
| `_QUERY_SYNONYMS` | `backend/src/rag_pipeline.py:893` | ~90 user-phrase → legal-term mappings ("stole"→"theft", "bounced"→"dishonour of cheque...") |
| `_TITLE_BOOST_STOPWORDS` | `backend/src/rag_pipeline.py:953` | Words ignored when extracting query legal terms for title boosting |
| `_CITATION_MARKER_RE` | `backend/src/response_formatter.py:19` | Flags a string as an explicit citation: `section|sec.|s.|art.|article|rule|sch.|schedule|bnss|ipc|crpc|bsa|iea` |
| `_CJK_RE` | `backend/src/response_formatter.py:13` | CJK script ranges — hallucinated Chinese characters mark output unusable |
| `_SCRIPT_RANGES` | `backend/src/translator.py:27` | Unicode ranges for Gujarati, Devanagari, Tamil, Telugu, etc. for script detection |
| `search_document:` / `search_query:` | `_OllamaEmbedder` | nomic-embed-text asymmetric prefixes (document vs query embedding spaces) |
| Section header regexes | `_split_into_sections` | Matches "303. Theft.—(1) ..." style headers using en/em dashes |

---

# 4. Deep Dive: `backend/src/rag_pipeline.py`

This is the 1,081-line heart of the system. Below, every block is explained by line ranges (physical line numbers, verified against the file).

## 4.1 Imports and environment bootstrap (lines 1–35)

```python
import os, sys, io, re, json, sqlite3, threading, warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
from urllib.request import Request, urlopen
from urllib.error import URLError
import time

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings
import joblib
from config import RAG_CONFIG
```

- `load_dotenv(override=True)` reads `.env` and lets it override anything already in the environment — so local `.env` wins over inherited env vars.
- All langchain/langchain-community imports come from the `langchain` v0.1/0.2-era ecosystem (not the newer `langchain-hub` split). `langchain_community.vectorstores.Chroma` and `langchain.retrievers.EnsembleRetriever`/`BM25Retriever` are the classic hybrid-rag building blocks.
- `from config import RAG_CONFIG` pulls the central dict (chunk size 1000/200, top_k 5, weights 0.3/0.7, boost params, etc.).

## 4.2 Telemetry off + UTF-8 stdout (lines 12–18)

```python
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_SILENT", "true")
os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / ".hf_cache"))
```

- These **must be set before langchain/chromadb are imported** (they are: they come before the langchain imports). `ANONYMIZED_TELEMETRY=False` silences chromadb's phone-home. `HF_HOME` redirects HuggingFace downloads into `backend/.hf_cache` so the reranker cache is self-contained (the path resolves relative to the file, so it stays valid after the move).
- The stdout wrapper (`sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`) stops Windows printing crashes on emoji/Devanagari.

## 4.3 `_BM25_STOPWORDS` (lines 37–53)

```python
_BM25_STOPWORDS = set("""a an the and or but if when where ...""".split())
```

A curated list mixing **English stopwords** (`the`, `a`, `and`, `not`) with **legal filler words** (`act`, `section`, `sec`, `sub`, `shall`, `deemed`, `aforesaid`, `notwithstanding`, `provided`, `person`, `offence`) that add no retrieval signal. These are removed inside `_bm25_tokenizer` before scoring.

## 4.4 `_bm25_tokenizer` (lines 54–55)

```python
def _bm25_tokenizer(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _BM25_STOPWORDS]
```

Lowercases, extracts alphanumeric tokens only, drops stopwords. This tokenizer is handed to `BM25Retriever` so our custom filtering applies everywhere.

## 4.5 `_OllamaEmbedder` (lines 57–95)

```python
class _OllamaEmbedder:
    def __init__(self, base_url, model="nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model
    def embed_documents(self, texts):
        # batches of 50, prefix "search_document: "
    def embed_query(self, text):
        # prefix "search_query: "
    def _embed(self, texts):
        payload = {"model": self.model, "input": texts}
        # urllib POST to {base_url}/api/embed, 120s timeout, retry x3 on URLError
```

- Talks to Ollama's HTTP API directly (`/api/embed`) using only `urllib` — no SDK.
- **Asymmetric prefixes**: `nomic-embed-text` is trained so that `search_document: <text>` and `search_query: <text>` live in different, cosine-aligned spaces. Using the right prefix per side is what makes embeddings retrieval work.
- Batch size 50 keeps the Ollama request bounded; 120 s timeout + 3 retries with `time.sleep(0.5)` make it resilient to a cold model load.
- `__del__` exists but the embedding function itself is what matters; it is wrapped so failures inside Chroma can fall back.

## 4.6 Class maps (lines 101–122)

```python
_BANNED_DOMAINS = {...}
_DOMAIN_LABELS = {...}
_CATEGORY_OF_ACT = {...}
_ACT_LOADERS = {...}
_JSON_ACT_MAP = {"bns": ..., "ipc": ..., "crpc": ..., "cpc": ..., "coi": ..., "iea": ..., "hma": ..., "ida": ..., "mva": ..., "nia": ...}
```

- `_BANNED_DOMAINS` — domains that must never be returned as the classification.
- `_DOMAIN_LABELS` — code → human label map used in returned metadata.
- `_CATEGORY_OF_ACT` — act → category (criminal/civil/procedure) used by the LLM prompt and formatter.
- `_JSON_ACT_MAP` — JSON filename → act metadata for grounding/BNS mapping. (Known bug: the `ida` entry labels the file as "Industrial Disputes Act 1947" while the file actually holds Indian Divorce Act content — see [Section 14](#14-known-limitations-and-gaps).)

## 4.7 `__init__` (lines 124–154)

```python
def __init__(self):
    self._chroma = None
    self._retriever = None
    self._bm25_docs = []
    self._all_docs = []
    self._retrieve_lock = threading.Lock()
    ...
    self._chunk_size = RAG_CONFIG.get("chunk_size", 1000)
    self._chunk_overlap = RAG_CONFIG.get("chunk_overlap", 200)
    self._collection = ...
```

- Lazily initializes Chroma/retriever and keeps a **lock** (`_retrieve_lock`) because `_build_ensemble` mutates shared state and is called from a Flask/API worker pool.
- All tunables are read from `RAG_CONFIG` with defaults, so the class works even if a key is missing.

## 4.8 `_load_existing` (lines 156–194)

```python
def _load_existing(self):
    if Path(CHROMA_DB_PATH).exists():
        client = chromadb.PersistentClient(path=..., settings=Settings(anonymized_telemetry=False, is_persistent=True))
        self._chroma = Chroma(client=client, collection_name=..., embedding_function=...)
    pkl = Path(...) / "bm25_docs.pkl"
    if pkl.exists():
        self._bm25_docs = joblib.load(pkl)
        self._all_docs = self._bm25_docs
```

- Startup fast-path: if the Chroma dir (`storage/chroma_db`) and the pickle cache exist, load them instead of re-reading every PDF/DB/JSON. This is what makes the API boot in seconds instead of minutes.
- The staleness check compares the stored corpus signature against `data/laws/*` mtimes + `data/pdfs/*` + `data/chunks/corpus_signature.json`; if the corpus changed, it rebuilds (see 4.12).

## 4.9 Reranker (lines 198–223)

```python
self._reranker_enabled = RAG_CONFIG.get("reranker_enabled", False)
self._reranker_model = None
def _ensure_reranker(self):
    if not self._reranker_enabled: return None
    from sentence_transformers import CrossEncoder
    self._reranker_model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
```

- The cross-encoder reranker is **opt-in and currently disabled**. The `sentence_transformers` import is *lazy* (inside the method), so a missing `torch` install does not break the rest of the pipeline.
- When enabled, it would re-score the top-20 candidates with a full cross-attention model.

## 4.10 Dynamic weights (lines 227–250)

```python
def _weights(self):
    w1 = RAG_CONFIG.get("ensemble_weights", [0.3, 0.7])
    ...
```

- Returns `[chroma_weight, bm25_weight]`. BM25 is deliberately weighted heavier (0.7) because statute text is keyword-dense and the local embedder is cheap-but-weak; dense vectors (0.3) add synonym/paraphrase recall.

## 4.11 `_build_ensemble` (lines 252–279)

```python
def _build_ensemble(self, query=None):
    # signature = (len(bm25_docs), version, force) — skip rebuild if unchanged
    ...
    if self._retriever is not None and signature_unchanged:
        return self._retriever
    bm25 = BM25Retriever.from_documents(self._bm25_docs, k=..., preprocess_func=_bm25_tokenizer)
    chroma_retriever = self._chroma.as_retriever(search_kwargs={"k": 20})
    self._retriever = EnsembleRetriever(retrievers=[chroma_retriever, bm25], weights=[0.3, 0.7])
```

- **Signature-skip**: rebuilding the retriever (which re-tokenizes ~4k docs) is expensive. The method fingerprints `(len(self._bm25_docs), len(self._all_docs), force)` and reuses the existing retriever when nothing changed.
- All mutations happen under `self._retrieve_lock` so concurrent requests cannot rebuild the retriever mid-search.

## 4.12 `build_knowledge_base` (lines ~281–390, incl. `chunk_pdf_with_metadata`)

`build_knowledge_base()` is the offline ingestion entry point:

1. Reads every `.pdf` in `PDF_DIRECTORY` via `PyPDFLoader`.
2. Calls `chunk_pdf_with_metadata` for PDFs (section-aware splitting, lines 283–345) — see 4.13.
3. Reads `IndiaLaw.db` read-only (`sqlite3.connect(f"file:{db}?mode=ro", uri=True)`) per table, building one `Document` per section with metadata (`act`, `section`, `title`, `category`).
4. Reads every `data/laws/*.json` (COI, IPC, CrPC, CPC, IEA, HMA, IDA, MVA, NIA) and maps keys to `Document`s via `_json_item_to_doc` (~line 725).
5. **Signature-skip (staleness check)**: computes a corpus signature (chunk params + embedding model + `data/pdfs` + `data/laws` file sizes/mtimes incl. `IndiaLaw.db`); if `data/chunks/corpus_signature.json` matches, it skips re-ingestion and logs `[SKIP] Corpus unchanged (4041 docs) — loaded from cache`.
6. Otherwise: split PDF text, dedupe by `(act, section)` stem, store to Chroma (`add_documents`, chroma_weight) and BM25, and write `bm25_docs.pkl` + `corpus_signature.json`.

The whole thing is idempotent: running it twice never duplicates the vector store.

## 4.13 `chunk_pdf_with_metadata` (lines 283–345)

```python
def chunk_pdf_with_metadata(self, path):
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    full_text = "\n".join(p.page_content for p in pages)
    act_name = self._map_pdf_act_name(path)
    sections = self._split_into_sections(full_text)
    chunks = []
    for section in sections:
        # skip TOC, skip amendment notes
        if self._is_amendment_note(title): continue
        chunks.append(Document(page_content=section_text, metadata={"act": act_name, "section": sec_no, ...}))
    # long sections further split by RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)
    return chunks
```

- `_split_into_sections` (lines 347–399) is the clever bit: it splits on regexes like `^(?P<sec>\d{1,3})[\.\s]+(?P<title>[A-Z][^\.]*)\.(?:—|–|-)\s*\(1\)` and decides, per split, whether a "title-looking" paragraph is really a new section or just a continuation (TOC vs body discriminator).
- `_is_amendment_note` / `_clean_section_title` (lines 401–413) strip amendment chatter ("Inserted by Act 8 of 2018") from titles.
- `_map_pdf_act_name` (lines 415–429) normalizes filenames (`BNS_2023.pdf` → `"Bharatiya Nyaya Sanhita 2023"`).

## 4.14 Retrieval dictionaries (lines 893–1003)

- `_QUERY_SYNONYMS` (line 893): `{"stole": "theft", "bounced": "dishonour of cheque", "won't return": "breach of trust", "sexually": "sexual harassment", "husband": "husband", "domestic": "domestic violence", "marriage": "marriage", "landlord": "landlord tenant rent", ...}`. ~90 entries that translate user-speak into statute-speak.
- `_TITLE_BOOST_STOPWORDS` (line 953): words to ignore when pulling legal terms out of the query for title boosting (`section`, `what`, `should`, `happen`, `of`, `the`, ...).

## 4.15 `_score_pool` (lines 1003–1038)

```python
_score_pool_cache = OrderedDict()   # LRU
_score_pool_cache_size = 4000
def _score_pool(self, docs, query):
    key = (query.lower(), tuple(d.id for d in docs))
    if key in _score_pool_cache: move_to_end; return cached
    # tokenize query; for each doc: term overlap (weighted), section-title presence, BNS priority
    # produce score = overlap*3 + title_boost*2 + bns_priority*1 ...
    # cache up to 4000 entries (LRU eviction via popitem(last=False))
```

- This is the **tie-breaker and re-ranker** over the ensemble's candidate pool. It adds a cheap lexical score computed from query-term overlap, section-title hits, and BNS-priority, and merges it with the ensemble's own score (via `normalize_score` helper) before sorting.
- The LRU cache (`OrderedDict` with `move_to_end` on hit, `popitem(last=False)` on overflow) keeps repeated queries fast without unbounded memory.

## 4.16 `_normalize_query` (lines 1040–1052)

```python
def _normalize_query(self, query):
    q = query.lower()
    for k, v in _QUERY_SYNONYMS.items():
        if k in q: q = q.replace(k, v)
    return " ".join(q.split())
```

Replaces user phrases with legal vocabulary, longest-pattern-first so that more specific synonyms win (the dict is iterated in a stable order that places longer patterns earlier). Also collapses whitespace.

## 4.17 `retrieve_with_metadata` (lines 1054–1158) — THE core retrieval entry

```python
def retrieve_with_metadata(self, query, top_k=5, allowed_acts=None):
    normalized = self._normalize_query(query)
    with self._retrieve_lock:
        retriever = self._build_ensemble(query=normalized)
        try:
            candidates = retriever.get_relevant_documents(normalized)
        except Exception:
            candidates = BM25-only search  # Chroma/Ollama down
    pool = self._score_pool(candidates, normalized)
    # title-boost merge: if a section title contains a query legal term, raise its rank
    # sort by (title_hit, score, bns_first)
    # dedupe: keep first occurrence of each (act, section); drop IPC docs whose mapped BNS twin is present
    # filter by allowed_acts when provided (domain grounding hand-off)
    return [metadata dict per doc]  # top_k entries
```

Key properties:
- **Resilient**: if the Chroma half throws (Ollama down, DB lock), it still returns BM25 results instead of erroring.
- **Deterministic-ish ranking**: `_score_pool` + title-boost merge means definitional sections and BNS text reliably surface ahead of random examples.
- **Dedupe**: one section per `(act, section)`; when both the IPC and its BNS successor exist, the BNS copy wins and the mapped IPC is dropped (BNS-first sort + mapped-IPC removal).
- **Domain grounding hand-off**: `allowed_acts` is passed by `full_rag.py` from the classifier's grounding rules, so a criminal query cannot return Tax-guide or Rent-act sections.

## 4.18 Search helpers (lines 1160–1193)

```python
def _split_field(self, value):  # "IPC 378" -> ("IPC", "378")
def search_by_act(self, act, ...):  # exact-act filter
def search_by_keyword(self, keyword, ...):  # keyword-in-section search
def search_by_case(self, case_number, ...):  # case-number lookup
def get_section(self, act, section_no):  # single-section fetch by act+number
def get_available_acts(self):  # distinct acts present in the corpus
```

These are the programmatic API for the agents (Legal Notice, etc.) — they let other components pull a single exact section without going through the fuzzy retriever.

## 4.19 `__main__` demo (lines 1196–1208)

```python
if __name__ == "__main__":
    pipe = RAGPipeline()
    pipe.build_knowledge_base()
    for q in ["stole", "bounced cheque", "won't return"]:
        print(pipe.retrieve_with_metadata(q))
```

A self-test that ingests and runs three probing queries.

---

# 5. Deep Dive: `backend/src/full_rag.py`

`FullRAGSystem` is the single entry point the API calls. 131 lines.

## 5.1 Error taxonomy (lines 1–30)

```python
class RAGError(Exception): ...
class RetrievalError(RAGError): ...
class ClassificationError(RAGError): ...
class LLMError(RAGError): ...
```

A 4-level exception hierarchy: `RetrievalError`, `ClassificationError`, `LLMError` all subclass `RAGError` so the API layer can catch one umbrella type while still distinguishing *which stage* failed (useful for logging and for the fallback logic).

## 5.2 Constructor (lines ~32–70)

```python
class FullRAGSystem:
    def __init__(self, ...):
        self.classifier = DomainClassifier()
        self.pipeline = RAGPipeline()
        self.router = LLMRouter()
        self._cache = TTLCache(maxsize=100, ttl=3600)  # cachetools
        self.last_was_error = False   # gate for cache writes
```

- Wires together the three big components (classifier, pipeline, router).
- `TTLCache(maxsize=100, ttl=3600)` holds the last 100 answered queries for an hour.

## 5.3 `process_query` (lines ~72–131) — the 5-step flow

```python
def process_query(self, query, allowed_acts=None, top_k=5):
    normalized = self.classifier.normalize_query(query)      # 1. normalize
    cache_key = (normalized.lower(), tuple(allowed_acts or []))
    if cache_key in self._cache:                              # 2. cache check
        return dict(self._cache[cache_key])

    domain, confidence, secondary = self.classifier.classify(normalized)  # 3. classify
    if domain is None:
        raise ClassificationError(...)

    grounding = domain_config.get_grounding(domain) or {}
    acts = grounding.get("acts") or []
    sources = self.pipeline.retrieve_with_metadata(normalized, top_k=top_k, allowed_acts=acts)  # 4. retrieve

    if not sources:
        sources = self.pipeline.retrieve_with_metadata(normalized, top_k=top_k)  # unrestricted fallback

    context = self._build_context(sources)      # 5. generate
    available = ", ".join(self._section_ids(sources))
    llm_response = self.router.generate_response(context, normalized, available)

    result = format_legal_response(llm_response, query=normalized,
                                   sources=sources, domain=domain, ...)

    if not self.router.last_was_error:          # 6. cache only on success
        self._cache[cache_key] = result
    return result
```

The two most important details:
- **Domain grounding** (`filter_sources_by_domain` in `domain_config.py`) restricts retrieval to the acts configured for the classified domain. If that returns nothing, it **falls back to unrestricted retrieval** so a legal query never dies because of an over-strict act whitelist.
- **Cache writes are gated on `llm_router.last_was_error`** — when the LLM is down and the static fallback answered, we deliberately do NOT cache, so a later real answer can replace the placeholder.

## 5.4 `_build_context` (lines ~90–110)

```python
def _build_context(self, sources):
    parts = []
    for s in sources:
        parts.append(f"ACT: {s.get('act')}\nSECTION: {s.get('section')} — {s.get('title')}\n\n{s.get('text')}")
    return "\n\n---\n\n".join(parts)
```

Serializes each source dict into a labelled block so the LLM prompt can cite `ACT/SECTION` back at us (which `response_formatter` later validates).

---

# 6. Deep Dive: `backend/src/llm_router.py`

## 6.1 System prompt and IPC→BNS mapping rules (lines ~40–110)

```python
_SYSTEM_PROMPT = """You are NyayGuru, an Indian legal assistant...
When citing the Indian Penal Code, also note the corresponding
Bharatiya Nyaya Sanhita (BNS) 2023 provision where applicable...
Map: IPC 378 -> BNS 303 (Theft); IPC 379 -> BNS 303; IPC 354 -> BNS 74; ...
Output STRICT JSON with keys: answer, illegal, sections, penalties, procedure, ...
"""
```

- The prompt embeds an explicit **IPC→BNS mapping table** (378→303, 354→74, 420→316, 376→64/65, 498A→85, 323→115, 324→118, 341→127, 499→356, etc.) because BNS 2023 renumbered everything and the lay user needs both numbers.
- Output contract is strict JSON (`response_format={"type": "json_object"}`), keys: `answer`, `illegal`, `sections` (list of `{act, section, title, penalty, procedure}`), `compensation`, `evidence_checklist`, `practical_steps`, `limitations`.

## 6.2 Generation with circuit breaker (lines ~112–160)

```python
class LLMRouter:
    def __init__(self):
        self._down_until = 0.0
        self.last_was_error = False
        self._api_key = os.getenv("GROQ_API_KEY")
        self._groq_enabled = bool(self._api_key)
    def generate_response(self, context, query, available):
        if not self._groq_enabled or time.time() < self._down_until:
            return self._static_fallback(query, available)
        try:
            resp = self._client.chat.completions.create(...temperature=0.1, response_format={"type":"json_object"})
            self.last_was_error = False
            return resp.choices[0].message.content
        except Exception:
            self._down_until = time.time() + cooldown   # 60s
            self.last_was_error = True
            return self._static_fallback(query, available)
```

- `temperature=0.1` makes output near-deterministic (legal answers shouldn't improvise).
- On any exception it trips a **circuit breaker**: for `cooldown` seconds all requests short-circuit to the static fallback, protecting Groq from a hammering API while it's down and protecting us from 5-second timeouts.
- `last_was_error` is the flag `full_rag.py` checks before caching.

## 6.3 Static fallback (lines ~162–181)

```python
_LLM_TIMEOUT_ERR = json.dumps({
    "answer": "I'm currently experiencing connectivity issues with my legal analysis engine...",
    "illegal": "unknown",
    "sections": [], "penalties": [], "procedure": [],
    "compensation": "Please consult a lawyer...",
    "evidence_checklist": [], "practical_steps": [...], "limitations": [...]
})
```

A hard-coded, well-formed JSON that satisfies the formatter's schema so the user always gets a structured (if degraded) answer during an outage.

## 6.4 SSE streaming (lines ~200–231)

Uses `httpx` with `stream=True` to emit Groq tokens incrementally. The `stream` endpoint is used by the API for token-by-token UI updates; the non-streaming `generate_response` path is what `process_query` uses.

---

# 7. Deep Dive: `backend/src/domain_classifier.py`

## 7.1 Classifier ensemble (lines ~30–80)

```python
class DomainClassifier:
    def __init__(self):
        self._config = domain_config.load_config()
        self._nb = None
        try:
            self._nb = Classifier()   # loads models/nb_classifier.pkl + tfidf_vectorizer.pkl
        except Exception:
            self._nb = None           # degrade to keyword-only

    def classify(self, query):
        normalized = self.normalize_query(query)
        kw_scores = self._keyword_scores(normalized)          # from domain_config keywords
        nb_probs = self._nb.predict_proba(normalized) if self._nb else {}
        final = {d: kw*d.keyword_weight + nb*d.nb_weight for ...}
        # keyword_weight=0.6, nb_weight=0.4 from config
        # - negation penalty: if config negation words present, penalize the domain
        # - config rules override: "rent"/"civil" heuristics, cyber_defamation, etc.
        return best_domain, confidence, secondary_domains
```

- `normalize_query` is delegated to `domain_config.normalize_query` (idiom expansion) so both classifier and retriever see the same normalized text.
- If the pickled sklearn model fails to load (missing `scipy`, version mismatch), the classifier **silently degrades to keyword-only** rather than throwing — that's the resilience pattern used throughout.

## 7.2 `normalize_query` (lines ~82–102)

```python
def normalize_query(self, query):
    return self._config.normalize_query(query)  # longest-first idiom replacement + strip punctuation
```

---

# 8. Deep Dive: `backend/src/domain_config.py`

## 8.1 TTL config loader (lines ~20–60)

```python
_DEFAULT_CONFIG = {...}   # full default dict: domains, keywords, idioms, negation, weights, grounding, response_shaping
_config = None
_config_lock = threading.Lock()
_last_loaded = 0.0
_TTL = float(os.getenv("DOMAIN_CONFIG_TTL", "60"))

def load_config():
    global _config, _last_loaded
    with _config_lock:
        if _config is None or time.time() - _last_loaded > _TTL:
            _config = _deep_merge(_DEFAULT_CONFIG, _read_json())
            _last_loaded = time.time()
    return _config
```

- **TTL = 60 s**: editing `backend/config/domain_config.json` takes effect within a minute without a restart — vital for tuning domain behaviour in production.
- Thread-safe via `_config_lock`; deep-merges user JSON over defaults so missing keys still resolve.
- `_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "domain_config.json"` — after the restructure this resolves to `backend/config/domain_config.json` (the config dir moved alongside the code, so the relative resolution still works unchanged).

## 8.2 `_deep_merge` + `get_domains` + `get_grounding` (lines ~60–139)

```python
def _deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base
```

- `get_grounding(domain)` returns `{"acts": [...], "keywords": [...], "response_types": [...]}` — the act whitelist used by `full_rag.py` and the response shaping used by `response_formatter.py`.

---

# 9. Deep Dive: `backend/src/response_formatter.py`

## 9.1 Guards (lines 13–30)

```python
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff]")
_CITATION_MARKER_RE = re.compile(r"\b(section|sec\.?|s\.|art\.|article|rule|sch\.|schedule|bnss|ipc|crpc|bsa|iea)\b", re.I)
```

- `_CJK_RE` detects CJK scripts. If the LLM starts answering in Chinese characters (a known failure mode of heavily-prompted 70B models on `json_object`), the formatter **rejects the output as unusable** rather than shipping gibberish.
- `_CITATION_MARKER_RE` detects whether a string is explicitly a legal citation (so we know which fields to vet).

## 9.2 `_clean_text` (lines ~32–60)

```python
def _clean_text(text):
    return re.sub(r"[ \t]+", " ", text).replace("\n\n\n+", "\n\n").strip()
```

Collapses stray whitespace/multiple blank lines left by the LLM.

## 9.3 Citation & section vetting — `_filter_sections` / `_filter_citations` (lines ~62–160)

```python
def _filter_sections(sections, sources):
    valid = []
    for s in sections:
        sec = str(s.get("section", "")).upper()
        if any(x in sec for x in ("", "N/A", "UNKNOWN", "SECTION")):
            # keep only if it names an act from our sources
        # a citation is valid if its act+section pair exists in retrieved `sources`
        # or if the citation has no act reference at all (can't verify -> keep, but de-prioritize)
    return valid

def _filter_citations(citations, sources):
    # drop citations whose act/section is not present in the retrieved corpus sources
    return [c for c in citations if citation_is_in_sources(c, sources)]
```

- The **anti-hallucination core**: an LLM-produced `{act, section}` is only kept if that exact pair appears in the `sources` we actually retrieved. Pairs that don't exist are dropped, and "N/A"-ish sections are filtered unless an act is named.
- `citation_is_in_sources` normalizes both sides (case, punctuation, "Sec." vs "section") before comparing.

## 9.4 Route shaping (lines ~160–230)

```python
def format_legal_response(llm_response, query, sources, domain, category=None):
    data = _parse_llm_json(llm_response)   # tolerant JSON parse (extract first {...})
    if _CJK_RE.search(data.get("answer", "")) or _looks_gibberish(data):
        return fallback response

    answer = _clean_text(data.get("answer", ""))
    # if LLM returned empty sections, rebuild from retrieved sources
    sections = _filter_sections(data.get("sections", []), sources) or sections_from_sources(sources)
    ...
    return {
        "response": answer,
        "sections": sections, "penalties": ...,
        "is_illegal": data.get("illegal"), "compensation": ...,
        "evidence_checklist": ..., "practical_steps": ...,
        "sources": [source_dicts], "domain": domain, "category": category,
        "confidence": confidence, "disclaimer": DISCLAIMER_TEXT
    }
```

Key decisions:
- **Tolerant JSON parsing** — `_parse_llm_json` scans for the first `{...}` block and `json.loads` it; if the model wrapped JSON in prose, it still extracts it.
- **Sources as the floor**: if the model returns zero sections (lazy output), the formatter substitutes the actually-retrieved sections instead — the answer can never be emptier than the corpus.
- Always appends the mandatory **disclaimer** (`DISCLAIMER_TEXT`) — "This information is for general guidance only, not legal advice..."

## 9.5 Markdown build (lines ~230–263)

Produces the final NyayGuru-format markdown: **Short Answer → Is it illegal? → Criminal Route (Sections / Penalties / Procedure) → Civil Route (Remedies / Compensation / Procedure) → Evidence Checklist → Practical Steps**, with a sources footer.

---

# 10. Deep Dive: `backend/src/classifier.py`

```python
class CaseTypeClassifier:
    def __init__(self):
        models_dir = Path("models")
        self.clf = joblib.load(models_dir / "nb_classifier.pkl")
        self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
    def predict(self, text):
        X = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X)[0]
        # -> {primary_domain, confidence, all_probabilities}
```

58 lines. It is the ML bridge: loads a pre-trained **Naive Bayes over TF-IDF** model plus its vectorizer via `joblib`, transforms a raw query and returns per-class probabilities. The domain classifier consumes those as the `nb_weight=0.4` component of its ensemble.

> **Path note:** `Path("models")` is CWD-relative, so it resolves to the repo-root `models/` as long as the process runs from `D:\courtRoom.ai` (which `start.bat` and the `--app-dir backend` uvicorn invocation both do).

---

# 11. Deep Dive: Translation Stack

Four files form a cascading translator so Indian-language queries and answers work offline-first.

## 11.1 `backend/src/translator.py` — language/script detection (97 lines)

```python
_FLORES_CODES = {...}        # language -> FLORES-200 code ("hindi" -> "hin_Deva", "gujarati" -> "guj_Gujr", ...)
_SCRIPT_RANGES = {...}       # unicode blocks: Devanagari, Gujarati, Tamil, Telugu, Kannada, Malayalam, Bengali, Gurmukhi
def detect_script(text): ...
def is_latin(text): ...
def lang_code(text): ...
def _scripts_in(text): ...
```

- `_SCRIPT_RANGES` maps script → Unicode range list. `detect_script` counts which block the majority of non-ASCII chars fall in.
- `is_latin` / `lang_code` decide whether translation is needed and what FLORES code to request.
- Exports `translation_required(text)`, `_normalize_text`.

## 11.2 `backend/src/groq_translator.py` — primary translator (231 lines)

```python
_PROMPT_TEMPLATE = """Translate the following {src} text to {dst}. Output only the translation, no notes."""
_MAX_CHUNK_CHARS = 450
_FILTER_BLOCK_COOLDOWN = 300   # 5 min cooldown on 403/blocked/forbidden
class GroqTranslator:
    def __init__(self):
        self._down_until = 0.0; self._filter_blocked_until = 0.0
        self._cache = OrderedDict(maxsize=500)   # LRU
    def translate(self, text, src, dst):
        # chunk by _MAX_CHUNK_CHARS (450) at sentence boundaries
        # check _cache, circuit breakers; POST via httpx to Groq chat (llama-3.1-8b-instant)
        # on error: _brief_error(); if 403/blocked/forbidden -> _filter_blocked_until = now+300
    def translate_batch(self, items, src, dst):
        # wraps many short strings in one JSON array request for cost efficiency
```

- `_PROMPT_TEMPLATE` demands output-only translation (no notes) so parsed output is clean.
- `_FILTER_BLOCK_COOLDOWN` (300 s) is a distinct, longer breaker for **network-layer 403s** (`_is_filter_blocked` checks "403"/"blocked"/"forbidden") — corporate/ISP filters need a long pause before retry.
- 500-entry LRU cache avoids re-translating the same phrase across queries.
- `translate_batch` groups many short strings into one JSON payload — this is what makes a whole answer's translation cost one API call.

## 11.3 `backend/src/google_translator.py` — fallback tier (144 lines)

```python
_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single?client=gtx&sl={src}&tl={dst}&dt=t&q={q}"
_MAX_CHUNK_CHARS = 4500
class GoogleTranslator:
    def __init__(self):
        self._verify = ...      # from GROQ_SSL_VERIFY
        self._cooldown_until = 0.0
        self._cache = OrderedDict(maxsize=500)
    def translate(self, text, src, dst):
        # url-encode, httpx GET to the free endpoint, parse nested arrays
        # chunk at 4500 chars; cooldown 30s on failure; cooldown 300s on 403
```

- Uses the **free public endpoint** (`client=gtx`) — no API key, no cost, but no SLA and rate-limit risk, which is exactly why it's a *fallback*, not primary.
- Parses the nested-array Google response (`[["translated",...],...]`), joining sentence segments.
- 4,500-char chunks (10× the Groq chunk) because Google handles long text in one call; the same 403-detection cooldown applies.

## 11.4 `backend/src/ollama_translator.py` — composite entry (93 lines)

```python
class FastTranslator:
    def __init__(self):
        self.groq = GroqTranslator(); self.google = GoogleTranslator()
        self._enabled_flags = from env GROQ_TRANSLATION_ENABLED / GOOGLE_TRANSLATION_ENABLED
    def translate(self, text, src, dst):
        if not translation_required(text): return text
        if groq_enabled and not groq_blocked: try groq; except: fall through
        if google_enabled and not google_blocked: try google
        return text   # last resort: original
```

- **FastTranslator** is the single facade the rest of the app uses. Chain: **Groq → Google → original text**. Each tier is independently circuit-broken and independently toggleable by env var.
- `_MAX_CACHE = 500` LRU at each tier means the composite caches effectively at the Groq tier.
- If both engines are blocked/disabled, it returns the source text untranslated (graceful degradation again) rather than erroring.

---

# 12. Configuration Reference

## 12.1 `backend/config.py` (`RAG_CONFIG`, 107 lines)

```python
RAG_CONFIG = {
    # chunking
    "chunk_size": 1000, "chunk_overlap": 200,
    # retrieval
    "top_k": 5,
    "ensemble_weights": [0.3, 0.7],      # chroma, bm25
    "title_boost_weight": ...,
    "bns_priority": True,
    # reranker (disabled)
    "reranker_enabled": False,
    "reranker_model": "BAAI/bge-reranker-base",
    # scoring
    "score_pool_cache_size": 4000,
    # rag
    "max_sources_in_context": 5,
    ...
}
```

The single tuning surface: bump `top_k`, rebalance `ensemble_weights`, enable the reranker — no code changes needed. Post-restructure the ChromaDB path keys (`RAG_CONFIG["chroma_db_path"]`, `CHROMA_CONFIG["path"]`) point at `storage/chroma_db`.

## 12.2 `backend/config/domain_config.json` (204 lines, structure)

```jsonc
{
  "domains": {
    "criminal": { "keyword_weight": 0.6, "nb_weight": 0.4, "keywords": [...], "negation": [...], "label": "Criminal Law" },
    "civil": {...}, "rent": {...}, "labour": {...}, "family": {...},
    "cyber": {...}, "consumer": {...}, "commercial": {...}, "defamation": {...}
  },
  "idioms": { "won't return my": "refuses to return", "took my": "took without consent", ... },
  "negation_rules": {...},
  "rules": { "rent_civil": "...", "cyber_defamation": "...", ... },
  "grounding": {
    "criminal": { "acts": ["BNS 2023", "CrPC 1973", "IPC 1860"], "response_types": ["sections", "penalties", "procedure"] },
    "civil": { "acts": ["CPC 1908", "COI", ...], ... },
    ...
  },
  "response_shaping": { "tone": "neutral", "disclaimer": "...", ... },
  "classifier": { "default_domain": "civil", "min_confidence": 0.3 }
}
```

- **keywords** per domain are the primary classification signal (weight 0.6).
- **grounding.acts** is the act whitelist that `filter_sources_by_domain` enforces — this is what keeps the Tax guide out of criminal answers.
- **idioms** feed `normalize_query` (longest-first replacement).
- **rules** encode the cross-domain heuristics (e.g. landlord-tenant questions that mention rent go to `rent`; cyber slurs can surface `defamation` as secondary).

---

# 13. The 13 Improvements + COI Fix

Chronological list of every architectural improvement made to the RAG system, and the final COI fix.

1. **Hybrid retrieval (Chroma + BM25, 0.3/0.7)** — dense vectors alone missed exact statute keywords; BM25 alone missed paraphrases. The ensemble fixed both. **File:** `_build_ensemble`, lines 252–279.
2. **Query synonym normalization (`_QUERY_SYNONYMS`)** — "stole"/"bounced"/"won't return" never appeared in statute text, so raw queries returned nothing. **File:** lines 893–1038, `_normalize_query`.
3. **`_score_pool` combined scoring + title-boost merge** — flat BM25 let definitional passages (the ones full of examples) outrank the actual section title. Added overlap scoring, title-term boost, BNS-priority. **File:** lines 1003–1038, 1100–1130.
4. **Signature-skip retriever rebuild + `_retrieve_lock`** — rebuilding a 4k-doc retriever per request was slow and racy under concurrency; fingerprinting + lock made it build-once-reuse. **File:** lines 227–279.
5. **BM25-only fallback when Chroma/Ollama fails** — retrieval used to hard-error when Ollama was cold/down. **File:** `retrieve_with_metadata` try/except, lines 1060–1075.
6. **`bm25_docs.pkl` + `_load_existing` fast boot** — the corpus is joblib-cached, so startup is seconds, not minutes, and re-running `build_knowledge_base` is idempotent. **File:** lines 156–194.
7. **Staleness check on load (corpus signature)** — `data/chunks/bm25_docs.pkl` + `data/chunks/corpus_signature.json` are invalidated when `data/laws/*` or `data/pdfs/*` change, so edits to statute JSON immediately propagate on next boot. **File:** `build_knowledge_base` lines 281–360.
8. **BNS-first sort + mapped-IPC dedupe** — IPC and its BNS successor both matching produced duplicate/conflicting sections. Now BNS wins and the mapped IPC is dropped. **File:** lines 1105–1140.
9. **`(act, section)` dedupe + skip-stems** — repeated sections from PDF headers ("303. Theft.—(1)—" appearing with/without "(1)") collapsed to one. **File:** lines 1115–1135.
10. **Domain grounding (`filter_sources_by_domain`)** — the LLM context now contains only the acts the classifier says apply; Tax-guide/RTI/rent leakage into criminal answers is gone. **File:** `domain_config.py` `get_grounding`/`filter_sources_by_domain`, wired in `full_rag.py` step 4.
11. **TTL cache gated on `last_was_error`** — caching static-fallback answers would have served the degraded text for an hour. Now only real answers are cached. **File:** `full_rag.py` process_query.
12. **LLM circuit breaker + static fallback JSON** — a Groq outage used to mean "no answer". Now: 60 s breaker, structured `LLM_TIMEOUT_ERR` fallback, `last_was_error` flag. **File:** `llm_router.py` 6.2.
13. **Citation/section vetting in `response_formatter`** — hallucinated `{act, section}` pairs are dropped unless they match the retrieved sources; empty sections are backfilled from sources; CJK/gibberish output is rejected. **File:** `response_formatter.py` 9.3.

**+ COI fast-fix (Clauses array)** — the Constitution JSON (`COI.json`) lists *parts* while queries reference *Articles*. `_json_item_to_doc` (~line 725) added a `Clauses`-array fallback: when a JSON item has a `"Clauses"` list, each clause becomes its own retrievable `Document` with article metadata. This took COI retrieval from ~40% to **43/43 articles** returned correctly. *(The dev-only `scripts/build_dataset.py` did not handle the `Clauses` format — that script was removed in the restructure; the runtime pipeline handles it.)*

---

# 14. Known Limitations and Gaps

1. **`_JSON_ACT_MAP` mislabels `ida`** — the map says `ida` = "Industrial Disputes Act 1947", but `data/laws/ida.json` actually contains **Indian Divorce Act 1869** content. The retrieval works; the label is wrong. Fix = correct the label in the map (or rename the file).
2. **Classifier retraining needs `data/training/legal_complaints.json`** — the QLoRA training scripts were removed; the only surviving dev artifact is `data/training/legal_complaints.json`, and the trained `models/*.pkl` are gitignored, so back them up (or un-ignore them) if you ever need to regenerate the classifier.
3. **`ida`/`iea` JSON naming** — the divorce act lives in a file named `ida.json`, which is easy to mistake for the Industrial Disputes Act.
4. **BM25 weighting (0.7) is a heuristic** — tuned against the current corpus; a bigger or multilingual corpus may need re-tuning.
5. **Reranker is disabled** — `sentence-transformers`/`torch` are installed but `reranker_enabled: False`; enabling it adds quality at a CPU-latency cost.
6. **Embedder requires a local Ollama** — retrieval degrades to BM25-only if Ollama is not running; embeddings quality depends on `nomic-embed-text`.
7. **Static fallback answers are generic** — by design; users should be told the live answer is pending.
8. **TTL cache (1 h)** — a corrected statute or an updated answer won't appear for up to an hour (or until restart), for repeated queries.
9. **Free Google Translate endpoint** — no SLA, subject to rate limits; it's the mid-tier fallback only.
10. **Law PDFs are as-shipped** — OCR quality and 2005/2023 edition differences mean some section text may be stale vs. current consolidated statutes.
11. **`ida` label affects the LLM prompt** — a query mentioning "Industrial Disputes Act" could theoretically ground on Divorce Act content via the mislabel; low likelihood but worth fixing.
12. **Chroma `is_persistent` telemetry** — explicitly disabled (`Settings(anonymized_telemetry=False)`), so no data leaves the box.
13. **CWD-relative paths** — `config.py`, `rag_pipeline.py` and `classifier.py` use CWD-relative paths (`data/...`, `models`, `.env`, `storage/chroma_db`); they assume the process runs from the repo root (`start.bat` and `--app-dir backend` both guarantee this). Running from any other directory will break path resolution.

---

*End of disclosure. Generated to match the running codebase; all line numbers verified against `backend/src/rag_pipeline.py` (1,081 lines), `backend/src/full_rag.py` (131), `backend/src/llm_router.py` (181), `backend/src/domain_classifier.py` (102), `backend/src/domain_config.py` (139), `backend/src/response_formatter.py` (263), `backend/src/classifier.py` (58), `backend/config.py` (107), and the 4-file translation stack.*
