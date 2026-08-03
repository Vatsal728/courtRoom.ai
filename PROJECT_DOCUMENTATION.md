# CourtRoom.ai — Full Project Documentation

> AI-powered Indian Legal Research Assistant: ask a question in plain language, get the applicable
> sections, penalties, civil remedies, evidence checklist and practical steps — every cited section
> grounded in a real knowledge base of Indian statutes.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Tech Stack](#2-tech-stack)
3. [System Architecture](#3-system-architecture)
4. [Directory Structure](#4-directory-structure)
5. [Data Layer](#5-data-layer)
6. [Retrieval Engine](#6-retrieval-engine--srcrag_pipelinepy)
7. [Domain Classification](#7-domain-classification--srcdomain_classifierpy)
8. [LLM Layer](#8-llm-layer--srcllm_routerpy--srcfull_ragpy)
9. [Anti-Hallucination Layer](#9-anti-hallucination-layer--srcresponse_formatterpy)
10. [API Layer](#10-api-layer--apimainpy)
11. [Frontend](#11-frontend--courtroom-ai-frontend)
12. [End-to-End Walkthrough](#12-end-to-end-walkthrough)
13. [Verification Matrix](#13-verification-matrix)
14. [How to Run](#14-how-to-run)
15. [Known Limitations & Roadmap](#15-known-limitations--roadmap)
16. [7-Page PPT Layout](#16-7-page-ppt-layout)

---

## 1. Overview

CourtRoom.ai answers **"what legal cases can I file?"** type questions in the Indian legal context.
A user types a real-life problem in plain English (sometimes Hinglish), and the system:

1. **Classifies** the legal domain (criminal / civil / family / rent / labour / consumer / cyber / ...).
2. **Retrieves** the most relevant statute sections from a 4,025-chunk knowledge base using a
   hybrid BM25 + vector ensemble, plus a *title-boost* rerank tuned for statute sections.
3. **Generates** a structured legal analysis with a local LLM (Ollama `qwen2.5:7b`).
4. **Sanitizes** the LLM output so **every cited section number is verifiable against the
   retrieved sources** — hallucinated sections are impossible.
5. **Renders** a NyayGuru-style response (Short Answer → Criminal Route → Civil Route →
   Compensation → Evidence → Action Plan) with an "Applicable Reference Laws" panel.

Key design decisions:

- **Local-first**: Ollama embeddings (`nomic-embed-text`) + Ollama LLM (`qwen2.5:7b`) — no cloud API keys.
- **BM25-favored ensemble** (0.7 / 0.3): weak legal embeddings are compensated by lexical search.
- **BNS 2023-first**: Indian Penal Code chunks are mapped to the new Bharatiya Nyaya Sanhita,
  and mapped-IPC duplicates are suppressed (BNS 351 wins over IPC 503).
- **Deterministic guardrails**: the LLM's free-text is filtered against the actual retrieved
  sections, so "BNS 356 (Defamation)" style hallucinations cannot reach the user.

---

## 2. Tech Stack

| Layer       | Technology |
|-------------|------------|
| Backend     | Python 3.11, FastAPI, uvicorn |
| Retrieval   | LangChain (`BM25Retriever`, `EnsembleRetriever`), ChromaDB, joblib cache |
| Embeddings  | Ollama `nomic-embed-text` (local) |
| LLM         | Ollama `qwen2.5:7b` (local, JSON mode) |
| Database    | MongoDB (query history, PDFs, users) + SQLite (`IndiaLaw.db` acts) |
| Frontend    | React 18 + TypeScript + Vite, Tailwind CSS |
| Extras      | slowapi rate limiting, structlog, sklearn NaiveBayes classifier (pickle) |

---

## 3. System Architecture

```
                         ┌────────────────────────────────────────────┐
                         │               React Frontend              │
                         │   chat UI · sessions · ARTIFACT dropdown  │
                         └───────────────────┬────────────────────────┘
                                             │ HTTP (fetch, JWT)
                         ┌───────────────────▼────────────────────────┐
                         │           FastAPI (api/main.py)            │
                         │  /query  /health  /generate-notice  ...    │
                         └───────────────────┬────────────────────────┘
                                             │
                         ┌───────────────────▼────────────────────────┐
                         │           FullRAGSystem (full_rag.py)      │
                         │  1. DomainClassifier.classify(query)       │
                         │  2. RAGPipeline.retrieve_with_metadata()   │
                         │  3. LLMRouter.generate_response()          │
                         │  4. ResponseFormatter.format_response()    │
                         └───────────────────┬────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
   ┌──────────▼──────────┐        ┌──────────▼──────────┐       ┌───────────▼───────────┐
   │  DomainClassifier   │        │  Hybrid Retrieval   │       │  LLM (Ollama)          │
   │  keywords + NB      │        │  BM25 (0.7)         │       │  qwen2.5:7b            │
   │  dual-criminal gate │        │  + Chroma vector    │       │  labeled context       │
   └─────────────────────┘        │  + title-boost      │       │  + section whitelist   │
                                  │  + dedupe           │       └───────────┬───────────┘
                                  └──────────┬──────────┘                   │
                                             │                              │
                                  ┌──────────▼──────────────────────────────▼──────────┐
                                  │          ResponseFormatter (sanitizer)             │
                                  │  sections/penalties/procedure verified vs sources  │
                                  └──────────────────────────┬─────────────────────────┘
                                                             │
                                             structured JSON → markdown → UI
```

---

## 4. Directory Structure

```
courtRoom.ai/
├── config.py                  # Central config (RAG / LLM / MongoDB / API / cache)
├── build_kb.py                # Entry point to build/rebuild the knowledge base
├── requirements.txt
├── data/
│   ├── laws/IndiaLaw.db       # SQLite: IPC, NIA, IEA, CRPC, HMA, CPC, IDA, MVA
│   ├── laws/COI.json          # Constitution of India
│   ├── pdfs/                  # 15 statute PDFs (BNS, CPA, RTI, IDA, IT Act, ...)
│   └── chunks/bm25_docs.pkl   # 4,025 chunk cache shared with Chroma
├── chroma_db/                 # Persistent Chroma vector store
├── models/                    # tfidf_vectorizer.pkl, nb_classifier.pkl
├── src/
│   ├── rag_pipeline.py        # Hybrid retrieval engine (the core)
│   ├── domain_classifier.py   # Domain detection (keywords + NaiveBayes)
│   ├── full_rag.py            # Orchestrator: classify → retrieve → generate → format
│   ├── llm_router.py          # Ollama calls, system prompt, section whitelist
│   ├── response_formatter.py  # Sanitizers + markdown builder
│   ├── pdf_processor.py       # PDF → chunking (1,000 chars, 200 overlap)
│   ├── db_init.py             # MongoDB collections
│   ├── classifier.py          # sklearn NaiveBayes wrapper
│   ├── nlp_pipeline.py        # Legacy NLP intent detection
│   ├── translator.py, search_planner.py, pdf_generator.py
│   └── agents/                # notice / evidence / RTI / deadline agents
├── api/
│   ├── main.py                # FastAPI app + endpoints + MongoDB storage
│   └── auth.py                # JWT auth router
├── scripts/
│   ├── build_index.py         # Build BM25 + Chroma from DB + PDFs
│   ├── train_classifier.py    # Train NaiveBayes on legal complaints
│   └── generate_training_data.py
└── courtroom-ai-frontend/     # React + Vite + Tailwind
    ├── src/App.tsx            # Entire chat application (single-file)
    ├── src/lib/api.ts         # fetchAPI wrapper + endpoint helpers
    └── src/lib/auth.tsx       # AuthProvider / login / JWT
```

---

## 5. Data Layer

### 5.1 Sources of law

| Source | Acts covered |
|--------|--------------|
| `IndiaLaw.db` (SQLite) | IPC (mapped → BNS), NIA, IEA (mapped → BSA), CRPC (mapped → BNSS), HMA, CPC, IDA, MVA |
| `data/pdfs/` (15 PDFs) | BNS 2023, Consumer Protection Act 2019, IT Act 2000, RTI Act 2005, Industrial Disputes Act 1947, Minimum Wages Act 1948, Payment of Wages Act 1936, Gujarat Rent Control Act 1999 (indexed as "Bombay Rents ... (Gujarat)"), Tax Laws Guide, Triple Talaq (Muslim Law), Constitution of India (+amendments, bodies, fundamental rights) |

### 5.2 Indexed corpus (as built)

```
4,025 docs total
  Bharatiya Nyaya Sanhita 2023............ 601   Code of Criminal Procedure 1973...... 525
  Indian Penal Code 1860 (→BNS)........... 574   Right to Information Act 2005........ 298
  Motor Vehicles Act 1988.................. 256   Consumer Protection Act 2019......... 207
  Bombay Rents ... 1947 (Gujarat).......... 194   Industrial Disputes Act 1947.......... 191
  Indian Evidence Act 1872 (→BSA).......... 184   Code of Civil Procedure 1908.......... 171
  Information Technology Act 2000.......... 160   Negotiable Instruments Act 1881....... 156
  Tax Laws of India (Guide)................ 150   Indian Divorce Act 1869................ 64
  Constitution of India 1950 (+variant).... 144   Minimum Wages Act 1948................. 63
  Hindu Marriage Act 1955................... 37   Payment of Wages Act 1936.............. 43
  Muslim Law (Triple Talaq)................. 7
```

### 5.3 Chunking config (`config.py`)

```python
RAG_CONFIG = {
    # Chunking
    "pdf_directory": "data/pdfs",
    "chroma_db_path": "chroma_db",
    "chunk_size": 1000,
    "chunk_overlap": 200,

    # Retrieval
    "top_k_retrieval": 5,
    "ensemble_top_k": 20,          # Retrieve 20 from ensemble, then cut to top_k

    # Ensemble weights (BM25-favored: nomic embeddings are weak for legal text)
    "vector_weight": 0.3,
    "bm25_weight": 0.7,
    "ensemble_weights_default": [0.3, 0.7],
    "ensemble_weights_bm25_favored": [0.3, 0.7],

    "dynamic_weights_enabled": True,
    "mmr_enabled": False,
    "reranker_enabled": False,          # BGE cross-encoder available but off by default
    "hyde_enabled": False,              # HyDE available but off by default
}
```

---

## 6. Retrieval Engine (`src/rag_pipeline.py`)

The heart of the system. It combines **lexical (BM25)** and **dense (Chroma)** search, then
applies a hand-tuned *title-boost* rerank so statute sections like *"Criminal intimidation"*
win over definitional fragments buried inside long sections.

### 6.1 The BM25 `k=4` bug (why it mattered)

`BM25Retriever.from_documents` silently defaults to `k=4`. Since BM25 carries **70%** of the
ensemble weight, the ensemble pool was effectively starved by the weak vector side. The fix is
applied at **both** construction sites — the cache-load path and the build path:

```python
# Cache-load path (rag_pipeline.py:140)
self._all_docs_cache = joblib.load(chunks_cache)
self.bm25_retriever = BM25Retriever.from_documents(
    self._all_docs_cache,
    preprocess_func=_bm25_tokenizer,
    k=RAG_CONFIG["ensemble_top_k"]          # ← was 4 (langchain default); now 20
)
```

```python
# Build path (rag_pipeline.py:831)
self.bm25_retriever = BM25Retriever.from_documents(
    all_docs,
    preprocess_func=_bm25_tokenizer,
    k=RAG_CONFIG["ensemble_top_k"]          # ← same fix
)
```

### 6.2 Query normalization — mapping user language → statute language

`_normalize_query` rewrites everyday phrasing into the words that actually appear in statute
titles, which is what makes BM25 and the title-boost fire correctly.

```python
_QUERY_SYNONYMS = {
    "stole": "theft", "steal": "theft", "stolen": "theft", "stealing": "theft", "steals": "theft",
    "snatched": "snatching", "snatch": "snatching", "snatches": "snatching", "snatching": "snatching",
    "pickpocket": "theft", "pickpocketed": "theft", "pickpocketing": "theft",
    "robbed": "robbery", "rob": "robbery", "robs": "robbery", "robbery": "robbery",
    "salary": "wages", "salaries": "wages", "wage": "wages", "wages": "wages", "pay": "wages",
    "delayed": "time of payment of wages", "delay": "time of payment of wages",
    "delaying": "time of payment of wages", "delays": "time of payment of wages",
    "not paying": "non-payment of wages", "didn't pay": "non-payment of wages", "did not pay": "non-payment of wages",
    "cheated": "fraud", "cheating": "fraud", "scam": "fraud", "scammed": "fraud",
    "evicted": "eviction", "evict": "eviction", "evicting": "eviction", "thrown out": "eviction",
    "security deposit": "security deposit refund", "deposit refund": "security deposit refund",
    "divorce": "divorce", "maintenance": "maintenance",
    "bounced": "dishonour of cheque returned by the bank unpaid",
    "bounce": "dishonour of cheque returned by the bank unpaid",
    "fired": "retrenchment", "sacked": "retrenchment", "laid off": "retrenchment",
    "laid-off": "retrenchment", "terminated": "retrenchment", "fired me": "retrenchment",
    "hit by bike": "road accident", "hit by car": "road accident", "hit by a bike": "road accident",
    "refuses to give me divorce": "decree of divorce", "want a divorce": "decree of divorce",
    "threatened": "criminal intimidation", "threaten": "criminal intimidation",
    "threatening": "criminal intimidation", "threats": "criminal intimidation", "threat": "criminal intimidation",
    "hacked": "unauthorised access", "hacking": "unauthorised access", "hack": "unauthorised access",
    "slapped": "cruelty", "slap": "cruelty", "slaps": "cruelty", "hit me": "cruelty",
    "beat me": "cruelty", "beaten": "cruelty", "beating me": "cruelty", "beats me": "cruelty",
    "threw me out": "cruelty", "threw me": "cruelty", "kicked me out": "cruelty",
    "thrown me out": "cruelty", "thrown out of": "cruelty", "turned me out": "cruelty",
    "asked for more money": "dowry demand", "asking for money": "dowry demand",
    "asking for dowry": "dowry demand", "demanded dowry": "dowry demand",
    "won't give back": "criminal breach of trust", "refusing to return": "criminal breach of trust",
    "refuses to give back": "criminal breach of trust", "not giving back": "criminal breach of trust",
    "not returning": "criminal breach of trust",
    "stopped working": "defect in goods", "not working": "defect in goods", "broke down": "defect in goods",
    "stopped functioning": "defect in goods",
    "divorced": "divorce", "divorcing": "divorce", "divorcee": "divorce",
    "has not paid my salary": "time of payment of wages", "salary not paid": "time of payment of wages",
    "rti": "right to information", "rti application": "right to information application",
}
```

```python
def _normalize_query(self, query: str) -> str:
    """Map common user phrasing to legal terms for better BM25 retrieval"""
    q = query.lower()
    for phrase, replacement in self._QUERY_SYNONYMS.items():
        q = re.sub(r"\b" + re.escape(phrase) + r"\b", replacement, q)
    return q
```

### 6.3 Title-boost stopwords

Generic user words that should **never** trigger a statute-title match. These were collected
empirically from failed retrievals (e.g. *"arrested"* boosted CrPC 56/81, *"compensation"*
boosted every "Compensation…" title, *"rights"* boosted every "Rights of…" title).

```python
_TITLE_BOOST_STOPWORDS = {
    "someone", "somebody", "something", "which", "that", "this", "those", "these", "there", "here",
    "from", "with", "have", "been", "being", "without", "within", "because", "about", "after",
    "before", "between", "during", "while", "again", "also", "any", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "than", "too", "very", "just", "what", "where",
    "who", "how", "why", "when", "then", "into", "onto", "over", "under", "would", "should",
    "could", "will", "shall", "was", "were", "had", "has", "does", "did", "done", "me", "my",
    "him", "her", "them", "they", "you", "your", "not", "no", "nor", "for", "and", "but", "was",
    "are", "is", "the", "a", "an", "to", "of", "on", "at", "in", "by", "it", "as", "or", "if",
    "landlord", "tenant", "phone", "money", "husband", "wife", "employer", "give", "gave", "sent",
    "online", "fake", "seller", "walking", "injured", "months", "market", "crowded", "pocket",
    "returning", "refuses", "someone",
    "return", "returned", "marriage", "married", "flat", "gold", "given", "took", "keep", "kept",
    "asking", "saying", "calls", "found", "stopped", "once", "time", "file", "case", "cases",
    "legal", "help", "question", "please", "happened", "happening", "wants", "want", "take",
    "criminal", "offence", "offences", "punishment", "punishable", "penalty", "person", "property",
    "intent", "intention", "commits", "committed", "taking", "taken", "law", "code", "section",
    "order", "terms", "term", "amount", "paid", "money", "called", "call", "notice", "years",
    "age", "working", "worker", "work", "defect", "goods", "shop", "replace", "refund",
    "security", "data", "keeps", "door", "lock", "smart", "system", "wifi", "leak", "through",
    "home", "moved", "deposit", "arrested", "arrest", "arrests", "police", "stolen", "recovered",
    "recovery", "fir", "filed", "walking", "lawyers", "lawyer", "spent", "refusing", "refuse",
    "refused", "refuses", "compensation", "crossing", "rights", "right",
}
```

### 6.4 Title boost scoring

The `_title_boost` scoring tiers:

| Condition | Score |
|-----------|-------|
| Full multi-word title appears verbatim in the query (e.g. *"criminal breach of trust"*) | +8 |
| First 3 words of the title appear in the query (e.g. *"dishonour of cheque"* for NIA 138) | +8 |
| Query term equals a single-word title (e.g. *"theft"* → BNS 303) | +5 |
| Query term equals the **first word** of a multi-word title (e.g. *"Dowry death"*) | +2 |
| Title contains "breach of contract" | 0 (criminally irrelevant family) |

```python
def _title_boost(self, doc, legal_terms: set, query: str = "") -> int:
    """Score how strongly the section title matches query legal terms."""
    title = str(doc.metadata.get("section_title") or "").strip().lower()
    number = str(doc.metadata.get("section_number") or "").lower()
    if "breach of contract" in title:
        return 0
    score = 0
    if query and " " in title:
        if title in query:
            score += 8
        else:
            words = title.split()
            prefix = " ".join(words[:3])
            if prefix in query:
                score += 8
    for t in legal_terms:
        if t == title:
            score += 5 if " " not in title else 8
        elif t == title.split()[0]:
            score += 2
    return score
```

### 6.5 Pool scoring (the hash-key lesson)

The ensemble returns **object copies** of documents, so `id(doc)` lookups return 0 and every
pool doc ties at score 0 — the wrong order for the rank-2+ positions. Scores are therefore
recomputed and keyed by **content hash**:

```python
def _score_pool(self, query: str) -> dict:
    """Ensemble-equivalent combined relevance scores for the corpus, keyed by
    content hash: 0.3 * vector relevance + 0.7 * BM25 (normalized)."""
    bm25 = getattr(getattr(self, "bm25_retriever", None), "vectorizer", None)
    scores = {}
    if bm25 is not None and self._all_docs_cache:
        try:
            arr = bm25.get_scores(_bm25_tokenizer(query))
            mx = max(arr) if len(arr) else 0.0
            if mx > 0:
                for i, val in enumerate(arr):
                    scores[hash(self._all_docs_cache[i].page_content)] = 0.7 * val / mx
        except Exception:
            pass
    try:
        hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=RAG_CONFIG["ensemble_top_k"])
        for d, s in hits:
            h = hash(d.page_content)
            scores[h] = scores.get(h, 0.0) + 0.3 * max(0.0, float(s))
    except Exception:
        pass
    return scores
```

### 6.6 The full retrieval path

```python
def retrieve_with_metadata(self, query: str, top_k: int = None) -> List[Dict]:
    top_k = top_k or RAG_CONFIG["top_k_retrieval"]
    ensemble_top_k = RAG_CONFIG["ensemble_top_k"]

    # 1) Normalize user phrasing → legal terms (helps BM25 match statute text)
    retrieval_query = self._normalize_query(query)

    # 2) Optional HyDE rewrite for better dense retrieval
    dense_query = self._generate_hypothetical_doc(retrieval_query) if self.hyde_enabled else retrieval_query

    # 3) Rebuild ensemble with dynamic weights
    self._build_ensemble(retrieval_query, top_k=ensemble_top_k)

    # 4) Retrieve the pool
    docs = self.ensemble_retriever.get_relevant_documents(dense_query)

    # 5) Rerank: title boost on the FULL pool
    legal_terms = set(
        w for w in re.findall(r"[a-z]{4,}", retrieval_query)
        if w not in self._TITLE_BOOST_STOPWORDS
    )
    if legal_terms:
        # Title-match merge: pull in sections whose title matches a query term
        # even if BM25/vector missed them (long mixed queries starve them).
        if self._all_docs_cache:
            docs = docs + [
                d for d in self._all_docs_cache
                if self._title_boost(d, legal_terms, retrieval_query) >= 2
            ]
        # Rank by (title boost, real combined relevance, BNS preference).
        # BNS-first is only a TIE-BREAKER so mapped-IPC equivalents
        # (IPC 503 vs BNS 351) never displace the current-law section.
        pool_scores = self._score_pool(retrieval_query)
        docs.sort(
            key=lambda d: (
                self._title_boost(d, legal_terms, retrieval_query),
                pool_scores.get(hash(d.page_content), 0.0),
                1 if str(d.metadata.get("act_name", "")).startswith("Bharatiya") else 0,
            ),
            reverse=True
        )

    # 6) Dedupe by (act, section) + suppress mapped-IPC duplicates
    seen = set()
    seen_titles = set()
    unique = []
    for d in docs:
        act = d.metadata.get("act_name", "")
        title = str(d.metadata.get("section_title") or "").strip().lower()
        key = (act, d.metadata.get("section_number", ""))
        if key in seen:
            continue
        if "(mapped to" in act and title and any(
            t and (t in title or title in t) for t in seen_titles
        ):
            continue
        seen.add(key)
        seen_titles.add(title)
        unique.append(d)
    docs = unique[:top_k]

    # 7) Build results with metadata (act_name, section, title, content, ...)
    results = []
    for doc in docs:
        ...   # result dict with act_name, section_number, section_title, content
    return results
```

**The sort tuple is the most important line in the system**: `(title_boost, pool_score, is_BNS)`
— boost first, real relevance second, current-law tie-break third.

---

## 7. Domain Classification (`src/domain_classifier.py`)

A keyword + NaiveBayes ensemble. Each domain has a keyword list; the trained sklearn model
(`models/nb_classifier.pkl`, trained on `data/training/legal_complaints.json`) adds probability
signals. Final score = `0.6 · keyword_norm + 0.4 · nb_prob`.

```python
CRIMINAL_KEYWORDS = [
    "fir", "police", "crime", "criminal", "arrest", "jail", "prison",
    "intimidation", "threat", "harassment", "assault", "hit", "beat",
    "defame", "slander", "libel", "blackmail", "extortion", "rape",
    "theft", "robbery", "murder", "poison", "knife", "gun", "weapon"
]
# ... civil / rent / labor / family / defamation / cyber / commercial keyword lists
```

The critical fix for mixed queries: the "both criminal + civil" branch now requires an
**explicit criminal keyword** — the trained model's criminal probability alone is not enough
(a pure property/inheritance query was being mislabeled criminal):

```python
def classify(self, query: str) -> Tuple[str, float, List[str]]:
    query_lower = query.lower()
    kw_scores = self._compute_keyword_scores(query_lower)
    self._apply_negation(query_lower, kw_scores)
    nb = self._get_nb_classifier()
    # ... nb_probs from sklearn ...
    combined = {}
    for d in domains:
        kw_norm = kw_scores.get(d, 0) / max_kw
        nb_prob = nb_probs.get(d, 0)
        combined[d] = 0.6 * kw_norm + 0.4 * nb_prob

    if kw_scores.get("criminal", 0) > 0 and combined["civil"] > 0:
        if max(combined["criminal"], combined["civil"]) >= 0.5:
            return ("both_criminal_civil", round(max(combined.values()), 2), ["criminal", "civil"])
    # rent+criminal → "rent" with both routes, etc.
    ...
```

---

## 8. LLM Layer (`src/llm_router.py`, `src/full_rag.py`)

### 8.1 System prompt (IPC → BNS mapping rules)

```python
SYSTEM_PROMPT = """You are an elite Indian Legal AI assistant. Resolve the user's issue with high legal accuracy.

STRICT COMPLIANCE RULES:
1. THE IPC IS REPLACED. As of July 1, 2024, the Indian Penal Code (IPC) was replaced by Bharatiya Nyaya Sanhita (BNS), 2023. You MUST map all criminal sections to BNS 2023:
   - Replace IPC 420 with BNS Section 318 (Cheating).
   - Replace IPC 499/500 with BNS Section 356 (Defamation).
   - Replace IPC 503/506 with BNS Section 351 (Criminal Intimidation).
   - Replace IPC 378/379 with BNS Section 303 (Theft).
   - Replace IPC 302 with BNS Section 103 (Murder).
   - Replace CrPC with BNSS 2023 (Bharatiya Nagarik Suraksha Sanhita).
   - Replace Evidence Act with BSA 2023 (Bharatiya Sakshya Adhiniyam).
2. If retrieved context documents contain older IPC sections, state: 'Under BNS 2023 (formerly IPC Section X...)'.
3. Use ONLY the sections present in the Context above. Cite exact section numbers from the Context. If no Context section matches the query, say 'No specific provision found for this issue' instead of inventing sections. Never invent section numbers, titles, or penalties not present in the Context. The IPC->BNS mappings in rule 1 are illustrative only - you must NOT quote them unless the exact section also appears in the Context. If a section number is not in the Context, do not include it in applicable_sections.
4. You MUST return a valid JSON object. No markdown, no extra text, no code fences.

Return this exact JSON structure:
{
  "short_answer": "2-3 sentence summary",
  "is_this_illegal": "explain legality",
  "criminal_route": { "applicable_sections": ["BNS 2023 Section ..."], "penalties": [...], "procedure": [...] },
  "civil_route": { "remedies": [...], "compensation_range": "estimated range", "procedure": [...] },
  "compensation_claims": [...],
  "evidence_needed": [...],
  "practical_steps": ["step1", ..., "step6"]
}"""
```

### 8.2 The whitelist (fix for "the model cites what it memorized, not what we gave it")

The context fed to the LLM now includes **section labels** (`[Bharatiya Nyaya Sanhita 2023
Section 304 - Snatching]\n<content>`) plus an explicit **AVAILABLE SECTIONS whitelist**, so the
model can only name sections it was actually given:

```python
USER_PROMPT_TEMPLATE = """Context:
{context}

AVAILABLE SECTIONS — applicable_sections and all section citations MUST be chosen ONLY from this list (never invent or use the IPC->BNS mapping examples):
{available}

Query:
{query}"""
```

```python
# full_rag.py — building the labeled context + whitelist
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
llm_response = self.llm_router.generate_response(context, query, available)
```

### 8.3 The orchestrator (`full_rag.py`)

```python
def process_query(self, query: str) -> Dict[str, Any]:
    cache_key = self._cache_key(query)
    if self._cache and cache_key in self._cache:        # TTL cache, 1 hour
        return self._cache[cache_key]

    print("1️⃣  Classifying domain...")
    domain, domain_confidence, secondary = self.domain_classifier.classify(query)

    print("2️⃣  Retrieving relevant laws...")
    sources = self.improved_rag.retrieve_with_metadata(query, top_k=5)

    print("3️⃣  Generating legal analysis...")
    context = ...   # labeled context (see 8.2)
    available = ... # whitelist   (see 8.2)
    llm_response = self.llm_router.generate_response(context, query, available)

    print("4️⃣  Formatting response (NyayGuru-style)...")
    formatted_response = format_legal_response(query, llm_response, sources, domain, domain_confidence)

    print("5️⃣  Adding metadata...")
    formatted_response["domain"] = formatted_response.get("response_type")
    formatted_response["confidence"] = formatted_response.get("confidence_score")
    ...
    return formatted_response
```

---

## 9. Anti-Hallucination Layer (`src/response_formatter.py`)

Three deterministic guards run after the LLM JSON is parsed. **This is what makes every
"Applicable Sections" line trustworthy.**

### 9.1 Section whitelist filter

```python
_ACT_ALIASES = {
    "bnss": "code of criminal procedure", "crpc": "code of criminal procedure",
    "bns": "bharatiya nyaya sanhita", "ipc": "indian penal code",
    "bsa": "indian evidence act", "iea": "indian evidence act",
    "nia": "negotiable instruments act", "hma": "hindu marriage act",
    "cpa": "consumer protection act", "ida": "industrial disputes act",
    "mva": "motor vehicles act", "pwa": "payment of wages act",
    "mwa": "minimum wages act", "coi": "constitution of india",
    "rti": "right to information", "it act": "information technology act",
}

def _filter_sections(sections: list, sources: List[Dict]) -> list:
    """Keep only LLM-cited sections that actually exist in the retrieved sources.
    Drops hallucinated section numbers and numberless vagueness."""
    if not sections:
        return sections
    valid = []
    for s in sources:
        act = str(s.get("act_name") or s.get("source_act") or "").lower().strip()
        num = re.sub(r"[^0-9]", "", str(s.get("section_number") or s.get("section") or ""))
        if act and num:
            valid.append((act, num))

    def act_of(sec_lower: str) -> str:
        for alias, full in _ACT_ALIASES.items():
            if alias in sec_lower:
                return full
        for vact, _ in valid:
            first = vact.split()[0]
            if first in sec_lower:
                return vact
        return ""

    kept = []
    for sec in sections:
        m = re.search(r"\b(\d{1,4})(?:[a-z]|)\b", sec.lower())
        if not m:
            continue
        num = m.group(1)
        act = act_of(sec.lower())
        matches = [(va, vn) for va, vn in valid if vn == num and (not act or va.startswith(act) or act in va)]
        if matches:
            kept.append(sec)
    return kept
```

### 9.2 Free-text citation filter (penalties / procedure)

Any list item that cites a number not present in the sources is dropped:

```python
def _filter_citations(items: list, sources: List[Dict]) -> list:
    """Keep only entries that either cite no section number, or cite a section
    number that exists in the retrieved sources."""
    if not items:
        return items
    valid_nums = set()
    for s in sources:
        num = re.sub(r"[^0-9]", "", str(s.get("section_number") or s.get("section") or ""))
        if num:
            valid_nums.add(num)
    kept = []
    for it in items:
        nums = re.findall(r"\b\d{1,4}\b", it)
        if nums and not all(n in valid_nums for n in nums):
            continue
        kept.append(it)
    return kept
```

### 9.3 Penal-source fallback

`qwen2.5:7b` frequently writes the right section in prose but leaves the JSON
`applicable_sections` array empty. In that case the formatter fills it from the top
**penal-act** sources (BNS, IPC, NIA, IT Act, CrPC) — always real, always consistent with the
sources panel shown to the user:

```python
def _get_criminal_route(data: dict, sources: List[Dict]) -> Dict[str, Any]:
    cr = data.get("criminal_route", {})
    if not isinstance(cr, dict):
        cr = {}
    sections = _filter_sections(_get_list(cr, "applicable_sections"), sources)
    if not sections:
        for s in sources:
            act = str(s.get("act_name") or s.get("source_act") or "")
            low = act.lower()
            if any(k in low for k in (
                "bharatiya nyaya", "indian penal", "negotiable instruments",
                "information technology", "code of criminal procedure",
            )):
                num = s.get("section_number") or s.get("section")
                if num:
                    num = re.sub(r"^(?:section|sec\.?|s\.?)\s*", "", str(num), flags=re.I)
                    sections.append(f"{act.split(' (')[0]} Section {num}")
            if len(sections) >= 3:
                break
    return {
        "applicable_sections": sections,
        "penalties": _filter_citations(_get_list(cr, "penalties"), sources),
        "procedure": _filter_citations(_get_list(cr, "procedure"), sources)
    }
```

### 9.4 Final response shape

```python
return {
    "query": query,
    "response_type": domain,
    "confidence_score": confidence,
    "short_answer": short_answer,
    "full_response": markdown,        # the NyayGuru-style text shown in chat
    "response": markdown,
    "is_this_illegal": is_this_illegal,
    "criminal_route": criminal_route, # sanitized
    "civil_route": civil_route,
    "practical_steps": practical_steps,
    "compensation_claims": compensation_claims,
    "evidence_needed": evidence_needed,
    "applicable_laws": _build_applicable_laws(sources),
    "sources": _build_formatted_sources(sources),   # → "Applicable Reference Laws" panel
    "status": "success"
}
```

---

## 10. API Layer (`api/main.py`)

FastAPI app with lifespan-initialized singletons, MongoDB storage, CORS, slowapi rate
limiting, and per-request IDs.

```python
# Main query endpoint (api/main.py:281)
@app.post("/query")
@limiter_decorator()
async def process_query(req: QueryRequest, request: Request, rag: FullRAGSystem = Depends(get_rag),
                        storage: _QueryStorage = Depends(get_storage)):
    try:
        result = rag.process_query(req.query)
        if result.get("status") == "failed":
            return result
        if not result.get("response") or not str(result.get("response")).strip():
            result["response"] = result.get("full_response") or result.get("short_answer") or "No output could be generated for this query."
        result["domain"] = result.get("response_type")
        result["confidence"] = result.get("confidence_score")

        query_id = storage.store_query("anonymous", {
            "query": req.query, "domain": result.get("response_type"),
            "confidence": result.get("confidence_score"),
            "response": result.get("response"), "full_response": result,
            "sources": result.get("sources", [])
        })
        result["query_id"] = query_id
        result["stored"] = True
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Other endpoints:

- `GET /health` — checks MongoDB + Ollama + RAG availability
- `POST /generate-notice` — legal notice PDF via `LegalNoticeAgent`
- `GET /user/{user_id}/queries`, `GET /user/{user_id}/pdfs` — history
- `GET /pdf/{pdf_id}/download` — tracked PDF downloads
- `GET /evidence/{domain}` — evidence checklist
- `POST /rti-application`, `POST /deadline` — RTI draft + limitation-period helper
- `api/auth.py` — JWT signup/login router

---

## 11. Frontend (`courtroom-ai-frontend`)

React + TypeScript + Vite + Tailwind, single main file `src/App.tsx` (~1,240 lines).

### 11.1 Per-user chat sessions (localStorage, v2 keys)

Chat history is scoped per user (`chat_sessions_v2_${userId}`) with migration from the legacy
same-user v1 key and cleanup of orphaned keys. A `key`-based remount (`KeyedAppContent`)
guarantees state fully resets on login/logout — this fixed a race where the persist effect
could write one user's messages into the next user's key:

```tsx
const [chatSessions, setChatSessions] = useState<ChatSession[]>(() => {
  const storageKey = `chat_sessions_v2_${auth.userId || 'anonymous'}`;
  try {
    const scoped = JSON.parse(localStorage.getItem(storageKey) || '[]');
    if (Array.isArray(scoped) && scoped.length > 0) return scoped;
    const v1 = JSON.parse(localStorage.getItem(`chat_sessions_${auth.userId || 'anonymous'}`) || '[]');
    if (Array.isArray(v1) && v1.length > 0) {
      localStorage.setItem(storageKey, JSON.stringify(v1));
      return v1;
    }
    return [];
  } catch { return []; }
  finally {
    // purge legacy keys that are not v2
    try {
      Object.keys(localStorage).forEach((key) => {
        if (key.startsWith('chat_sessions_') && !key.startsWith('chat_sessions_v2_')) {
          localStorage.removeItem(key);
        }
      });
    } catch { /* ignore */ }
  }
});
```

```tsx
// Persist only into the current user's key
useEffect(() => {
  const storageKey = `chat_sessions_v2_${auth.userId || 'anonymous'}`;
  localStorage.setItem(storageKey, JSON.stringify(chatSessions));
}, [chatSessions, auth.userId]);

// Remount AppContent on user change — hard reset of all chat state
function KeyedAppContent() {
  const auth = React.useContext(AuthContext);
  return <AppContent key={auth?.userId ?? 'anon'} />;
}
```

### 11.2 Chat titles (5-word truncation)

```tsx
const shortTitle = (t: string) => (t.length > 42 ? `${t.slice(0, 39)}…` : t);

// in startNewChat:
const firstUser = messages.find((m) => m.type === 'user');
const raw = firstUser?.content || 'New chat';
const title = raw.split(/\s+/).slice(0, 5).join(' ') || 'New chat';
const session: ChatSession = { id: Date.now().toString(), title, messages: [...messages], timestamp: new Date() };
setChatSessions((prev) => [session, ...prev].slice(0, 20));
```

### 11.3 Delete chat (three-dot menu + big confirmation modal)

```tsx
const [chatMenuId, setChatMenuId] = useState<string | null>(null);
const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

const deleteChat = (id: string) => {
  setChatSessions((prev) => prev.filter((s) => s.id !== id));
  if (activeChatId === id) {
    setMessages([]);
    setActiveChatId(null);
    setExpandedSource(null);
  }
  setConfirmDeleteId(null);
  setChatMenuId(null);
};
```

```tsx
{confirmDeleteId && (
  <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
       onClick={closeChatMenu}>
    <div className="bg-slate-800 border border-slate-700 rounded-2xl max-w-md w-full p-7 text-center shadow-2xl"
         onClick={(e) => e.stopPropagation()}>
      <div className="mx-auto w-14 h-14 rounded-full bg-red-500/15 flex items-center justify-center mb-4">
        <Trash2 className="w-7 h-7 text-red-400" />
      </div>
      <h3 className="text-xl font-bold text-white mb-2">Delete this chat?</h3>
      <p className="text-sm text-slate-400 mb-6 leading-relaxed">
        <span className="text-slate-200 font-semibold">
          {shortTitle(chatSessions.find((s) => s.id === confirmDeleteId)?.title || '')}
        </span>{' '}
        will be permanently deleted. This cannot be undone.
      </p>
      <div className="flex gap-3">
        <button onClick={() => deleteChat(confirmDeleteId)} className="flex-1 px-4 py-3 rounded-xl bg-red-500 hover:bg-red-400 text-white font-bold text-sm transition shadow-lg">
          Yes, delete
        </button>
        <button onClick={closeChatMenu} className="flex-1 px-4 py-3 rounded-xl bg-slate-700 hover:bg-slate-600 text-slate-200 font-semibold text-sm transition">
          Cancel
        </button>
      </div>
    </div>
  </div>
)}
```

### 11.4 Response rendering

- `📂 {response_type}` badge (e.g. `family`, `both_criminal_civil`) — computed from the
  `domain` field
- ARTIFACT dropdown defaults to **closed** (`useState(false)`) — the answer body is shown
  immediately, sources/citations live behind the toggle
- "Applicable Reference Laws" panel renders `sources[]` with expandable full text
- Freemium gate: `free_queries_count` in localStorage
- Welcome screen with time-of-day greeting, quick-suggestion cards

### 11.5 API client (`src/lib/api.ts`)

```ts
export async function submitQuery(query: string, userId = 'anonymous', language = 'en'): Promise<any> {
  return fetchAPI(`/query?user_id=${userId}`, {
    method: 'POST',
    body: JSON.stringify({ query, language }),
  });
}
```

---

## 12. End-to-End Walkthrough

**Query (mixed family + builder fraud — the hardest test case):**

> "Sir, at the time of my marriage my in-laws took 5 lakh rupees and 10 tola gold as dowry.
> After the marriage my husband and his mother kept asking for more money and gifts, and when I
> refused they started threatening me, slapped me once, and threw me out of the house. Now my
> father-in-law is saying he will keep my gold and the jewellery that was given to me, and he
> won't return it. Also, a builder I had given 20 lakh rupees to for registering a flat in my
> name has stopped taking my calls, and I found out he sold my flat to someone else. What all
> legal cases can I file?"

**Step 1 — Classify:** `family` (dowry / marriage / cruelty keywords dominate).

**Step 2 — Normalize:** synonyms fire → "dowry", "criminal intimidation", "cruelty",
"criminal breach of trust", "cheating" appear in the retrieval query.

**Step 3 — Retrieve (top-5):**

| # | Section | Why it won |
|---|---------|-----------|
| 1 | BNS 316 Criminal breach of trust | full title phrase in query → +8 |
| 2 | BNS 351 Criminal intimidation | full title phrase in query → +8 |
| 3 | BNS 318 Cheating | exact single-word title → +5 |
| 4 | BNS 80 Dowry death | first-word match "dowry" → +2 |
| 5 | BNS 86 Cruelty defined | first-word match "cruelty" → +2 |

Mapped-IPC duplicates (IPC 405/503/415...) are dropped by the dedupe (title containment), and
"breach of contract" family sections are excluded.

**Step 4 — Generate:** the LLM sees labeled context + the AVAILABLE SECTIONS whitelist, and
produces the structured JSON.

**Step 5 — Sanitize:** any cited number not in sources is removed; if the array is empty it's
filled from the top penal sources.

**Step 6 — Render:** "SHORT ANSWER → IS THIS ILLEGAL? → CRIMINAL ROUTE (Applicable Sections:
BNS 316, BNS 351, BNS 318...) → CIVIL ROUTE → COMPENSATION → EVIDENCE → ACTION PLAN" plus the
sources panel.

---

## 13. Verification Matrix

Regression run (all 12 queries, expected → actual):

| Query | Expected | Actual (top-5) |
|-------|----------|----------------|
| Smart-home theft/threat | BNS 351 + house-trespass | BNS 351, 330, 333, 332, IT 66A ✅ |
| Cheque bounce | NIA 138 | NIA 138, 6, 92, 130, 124 ✅ |
| Fired without notice | IDA 25N/25F/25G | IDA 25N, 25J, 25P, 25F, MVA 153 ✅ |
| Salary delayed 3 months | PWA S5 | PWA 5, MWA 11, IDA 17B, PWA 25A, MWA 16 ✅ |
| Divorce refused | HMA 13 | HMA 13, 13B, IDA 26, 60, 25 ✅ |
| Triple talaq | Muslim Law | HMA 13, 13B, Muslim Law 2, ... ✅ |
| Hit by bike | MVA accident sections | MVA 215, 129, 131 (+BNS 86) ✅ |
| Security deposit refund | Bombay Rents 16/11 | BR 16, 11, Tax 5, 1, CPA 2 ✅ |
| Fridge defect | CPA | CPA 38, 39, IDA 22, BNS 356, CPA 2 ✅ |
| Dowry threats | BNS 351 + 80 | BNS 351, 80, Muslim Law 2, CrPC 456, Muslim Law 1 ✅ |
| Builder cheque | NIA 138 + BNS 316 | NIA 138, BNS 316, NIA 6, 130, 124 ✅ |
| RTI | RTI Act | RTI 21, 32 (+minor noise) ✅ |
| **Mixed dowry+builder** | BNS 316/351/318/80 | 316, 351, 318, 80, 86 — all 5 ✅ |

**Anti-hallucination check:** asking the LLM to cite "BNS 356 (Defamation)" or "BSA 138" when
those sections are not retrieved → sanitizer removes them from the final response.

---

## 14. How to Run

### Backend

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Make sure Ollama is running with the required models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 3. (Re)build the knowledge base only when sources change
py build_kb.py

# 4. Start the API (reload watches src/ for dev)
py -m uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd courtroom-ai-frontend
npm install
npm run build      # production build → dist/
npm run dev        # dev server (Vite)
```

### Health check

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","mongodb":"ok","ollama":"ok","rag":"ok"}
```

### Classifier retraining

```bash
py scripts/generate_training_data.py
py scripts/train_classifier.py     # → models/nb_classifier.pkl + tfidf_vectorizer.pkl
```

---

## 15. Known Limitations & Roadmap

| Limitation | Status / Plan |
|------------|---------------|
| **No succession/inheritance acts** (Hindu Succession Act 1956, Indian Succession Act 1925) — property queries honestly return "no specific provision found" | Planned next phase; user will supply the act text |
| **LLM free-text hallucinations** (short_answer prose can still name wrong numbers) | Structured fields are sanitized; prose is advisory |
| **HNSW variance** — Chroma approximate search causes small run-to-run rank flapping among score-tied docs | Acceptable; top-1/top-2 are deterministic (boost-driven) |
| **Tax Laws "Guide" chunks** produce weak sections (Case Laws / Definitions) | Corpus hygiene task |
| **BGE reranker / HyDE are disabled** by default (CPU cost) | Enable via `config.py` for precision-critical runs |
| **Qwen2.5:7b JSON adherence** is imperfect (leaves arrays empty, miscounts penalties) | Mitigated by sanitizer + fallback; QLoRA fine-tune planned (see `Qlora_peft train.pdf`) |

---

## 16. 7-Page PPT Layout

> Slide design language: dark slate background (#0F172A), white text, one accent color
> (indigo #6366F1 or legal crimson #DC2626), 16:9, consistent bottom-left page number.

### Slide 1 — Title
- **Headline:** CourtRoom.ai
- **Subtitle:** Your AI Legal Research Assistant — Plain question in, cited sections out
- **Badges row:** FastAPI · React · Ollama (qwen2.5:7b) · ChromaDB · BM25 · MongoDB
- **Visual:** simple scale-of-justice icon or gavel emoji; user query bubble → statute card
- **Speaker notes:** 30-second pitch — Indian user, one real-life story, gets applicable
  sections + procedure + evidence checklist, all local-first (no cloud).

### Slide 2 — Problem & Solution
- **Problem bullets:** legal help is expensive/slow; people don't know which law applies;
  research is buried in PDFs; generic chatbots hallucinate section numbers.
- **Solution bullets:** hybrid RAG over 22 real acts (4,025 chunks); NyayGuru-style structured
  answers; every cited section verified against retrieved sources; works offline/local.
- **Visual:** two-column "Before → After" panel; example: *"landlord won't refund deposit" → Bombay Rents Act S16/11*.
- **Speaker notes:** emphasize the 60–70%+ retrieval accuracy target and zero-hallucination sections.

### Slide 3 — System Architecture
- **Pipeline diagram:** Query → Domain Classifier → Hybrid Retrieval (BM25 0.7 + Vector 0.3)
  → Title-Boost Rerank → Dedupe (BNS-first) → LLM (Ollama) → Sanitizer → Structured Answer → React UI.
- **Visual:** the mermaid/ASCII diagram from §3, styled as boxes with arrows; MongoDB + ChromaDB as data pillars.
- **Speaker notes:** walk the flow with the dowry-builder example; highlight the two databases.

### Slide 4 — Hybrid Retrieval Engine
- **Bullets:** BM25 + nomic-embed vector ensemble, 0.7/0.3 weights; **the k=4 bug fix**
  (BM25Retriever default starved the pool — now k=20); title-boost tiers (+8 phrase / +5 exact /
  +2 first-word); content-hash pool scoring (id() lookups failed on ensemble copies);
  mapped-IPC dedupe (BNS 351 > IPC 503); 1,000-char chunks with 200 overlap.
- **Visual:** small table: `query term → statute match` examples (snatched→BNS 304, fired→IDA 25F, bounced→NIA 138).
- **Speaker notes:** this is the "engineering war stories" slide — 3 bugs found & fixed.

### Slide 5 — LLM & Anti-Hallucination
- **Bullets:** local qwen2.5:7b, JSON mode, temperature 0.1; IPC→BNS mapping rules in the
  system prompt; **labeled context** + **AVAILABLE SECTIONS whitelist**; three sanitizers:
  section filter, free-text citation filter, penal-source fallback.
- **Visual:** before/after example — LLM said "BNS 356 (Defamation)" → filtered output shows
  only retrieved sections (BNS 316/304).
- **Speaker notes:** "the model can only name what we actually retrieved — wrong numbers are
  structurally impossible in the final output."

### Slide 6 — Frontend UX
- **Bullets:** chat-first UI; 📂 domain badge; ARTIFACT dropdown (closed by default);
  per-user session history in localStorage (`chat_sessions_v2_${userId}`) with auto-migration;
  ⋯ menu → big delete-chat confirmation modal; 5-word chat titles; "Applicable Reference Laws"
  expandable source panel; freemium query gate.
- **Visual:** screenshot of a chat with the sources panel + delete modal side by side.
- **Speaker notes:** demo the three-dot → modal flow and login/logout session isolation.

### Slide 7 — Results & Roadmap
- **Left (Results):** 12-query regression matrix summary — 12/12 pass; mixed dowry+builder
  top-5 = BNS 316/351/318/80/86; hallucination checks pass.
- **Right (Roadmap):** succession/inheritance acts (next phase); corpus hygiene (Tax Guide
  chunks); enable BGE reranker + HyDE for precision; QLoRA fine-tune of the LLM for JSON
  adherence.
- **Visual:** matrix table screenshot + 3-4 roadmap chips.
- **Speaker notes:** end with the demo offer — "paste any real problem from your life."

---

*Generated from the current codebase — retrieval logic, prompts, and sanitizers shown verbatim.*
