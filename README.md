# 🏛️ courtRoom.ai

An AI-powered Indian legal assistant that helps citizens understand their rights, navigate legal procedures, and draft legal documents — all through natural conversation in 11 Indian languages.

---

## ✨ What It Does

- 💬 **Ask any legal question** in English or 11 Indian languages and get cited, plain-language answers referencing the exact sections of Indian law
- 📄 **Generate legal notices** (PDF) with your name, address, and issue details filled in
- 📝 **Draft RTI applications** ready to copy and file
- 🗂️ **Case strategy planning** with step-by-step roadmaps and success-chance estimates
- 🔍 **Document auditing** — upload a contract or agreement and get a clause-by-clause risk analysis
- ✅ **Evidence checklists** tailored to your situation
- 📁 **Case management** — track cases, upload evidence, and maintain a persistent workspace
- 🌐 **Multilingual** — ask in Hindi, Gujarati, Tamil, or 8 other languages and get answers in your language

---

## 🧠 AI Models Used

| Component | Model | Provider | Purpose |
|-----------|-------|----------|---------|
| **Legal analysis (primary)** | `llama-3.3-70b-versatile` | Groq Cloud | Generating legal responses, JSON extraction, tool routing |
| **Legal analysis (fallback)** | `qwen2.5:3b` | Ollama (local) | Local fallback when Groq is unavailable |
| **Translation** | `llama-3.1-8b-instant` | Groq Cloud | Translating queries to/from English |
| **Embeddings (user docs)** | `gemini-embedding-001` | Google AI | Vectorizing uploaded user documents |
| **Embeddings (master KB)** | `nomic-embed-text` | Ollama (local) | Vectorizing the legal knowledge base |
| **Reranker (optional)** | `BAAI/bge-reranker-base` | HuggingFace | Cross-encoder reranking for higher retrieval precision |
| **Domain classification** | Naive Bayes + TF-IDF | scikit-learn (local) | Classifying queries into legal domains |
| **Fact extraction** | Regex + keyword matching | Local | Extracting structured facts from user queries for tool execution |

---

## 🏗️ Architecture

```
courtRoom.ai/
├── backend/                     # FastAPI + Python 3.11+
│   ├── api/
│   │   ├── main.py              # REST endpoints, SSE streaming, multi-turn tool state
│   │   └── auth.py              # JWT authentication
│   ├── src/
│   │   ├── agent.py             # AI tool router + 4 agent tools + multi-turn merge
│   │   ├── llm_router.py        # Groq → Ollama fallback chain for generation
│   │   ├── full_rag.py          # Ensemble RAG (BM25 + vector + optional reranker)
│   │   ├── fact_extractor.py    # Domain + issue classification from queries
│   │   ├── embedding_provider.py # Google Gemini or Ollama embedding provider
│   │   ├── groq_translator.py   # Groq-powered translation
│   │   ├── google_translator.py # Google Translate fallback
│   │   ├── agents/
│   │   │   ├── notice_agent.py  # Legal notice PDF generation (ReportLab)
│   │   │   ├── rti_agent.py     # RTI application drafting
│   │   │   ├── strategy_agent.py # Case strategy planning
│   │   │   ├── audit_agent.py   # Document clause audit
│   │   │   └── evidence_agent.py # Evidence checklist generation
│   │   └── ...
│   ├── config/
│   │   ├── domain_config.json   # Legal domain keywords, missing-fact rules
│   │   ├── compensation_rules.json
│   │   └── document_audit_rules.json
│   └── requirements.txt
├── frontend/                    # React 19 + TypeScript + Vite 8 + Tailwind CSS 4
│   └── src/
│       ├── App.tsx              # Main app — chat, artifacts, documents, PDF viewer
│       └── lib/
│           ├── api.ts           # API client (stream, query, upload, PDF endpoints)
│           ├── auth.tsx         # Auth context (login, register, JWT)
│           ├── language.ts      # Language definitions and translations
│           └── types.ts         # TypeScript interfaces
├── data/
│   └── laws/                   # Indian law JSON database (BNS, IPC, CPC, CrPC, etc.)
├── storage/                    # ChromaDB vector databases (gitignored)
├── .env                        # Environment variables (gitignored)
└── start.bat                   # One-click launcher for backend + frontend
```

---

## ⚡ How the RAG Pipeline Works

1. **Query classification** — Your question is classified into a legal domain (criminal, civil, rent, labor, consumer, cyber, RTI, family) with a confidence score
2. **Fact extraction** — Structured facts (dates, amounts, parties, issue type) are extracted to detect missing information and guide tool selection
3. **Tool routing** — If the query matches a tool (legal notice, RTI, case strategy, document audit), it's routed there first via regex + LLM tool-calling
4. **Ensemble retrieval** — For general legal questions, a hybrid BM25 + vector search (weighted 70/30 in favor of BM25 for legal text precision) retrieves the most relevant sections of Indian law
5. **Optional reranking** — BGE cross-encoder reranker can be enabled for higher-precision retrieval
6. **Response generation** — Groq (`llama-3.3-70b-versatile`) generates the legal analysis, with local Ollama (`qwen2.5:3b`) as automatic fallback
7. **Translation** — If your query was in a non-English language, the response is translated back via Groq (`llama-3.1-8b-instant`) with Google Translate as fallback

---

## 🌐 Supported Languages

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | `en` | Kannada | `kn` |
| Hindi | `hi` | Bengali | `bn` |
| Gujarati | `gu` | Malayalam | `ml` |
| Marathi | `mr` | Punjabi | `pa` |
| Tamil | `ta` | Urdu | `ur` |
| Telugu | `te` | | |

---

## 🤖 AI Agents (Tools)

| Agent | Trigger | What It Does |
|-------|---------|--------------|
| 📄 **Legal Notice** | "legal notice", "demand letter" | Generates a formal legal notice PDF with sender/recipient details, issue description, and applicable law |
| 📝 **RTI Application** | "rti", "right to information" | Drafts a Right to Information application with applicant details and information sought |
| 🗂️ **Case Strategy** | "strategy", "what should I do", "chances" | Provides step-by-step case strategy, roadmap, and success-chance analysis |
| 🔍 **Document Audit** | "audit", "review", "check document" | Analyzes a contract/agreement clause-by-clause, flags risks and missing protections |
| ✅ **Evidence Checklist** | "evidence", "what proof do I need" | Generates a tailored evidence checklist based on your legal situation |

All agents support **multi-turn conversation** — if the bot needs more details, it asks targeted follow-up questions and remembers your answers across messages.

---

## 📦 Legal Database

The system ships with a comprehensive database of Indian law in structured JSON format:

| Act | Description |
|-----|-------------|
| **BNS 2023** | Bharatiya Nyaya Sanhita (replaced IPC) |
| **IPC** | Indian Penal Code (legacy reference) |
| **CrPC** | Code of Criminal Procedure |
| **CPC** | Code of Civil Procedure |
| **HMA** | Hindu Marriage Act |
| **IDA** | Industrial Disputes Act |
| **IEA** | Indian Evidence Act |
| **MVA** | Motor Vehicles Act |
| **NIA** | Negotiable Instruments Act |
| **COI** | Constitution of India |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- MongoDB running on `localhost:27017`
- Ollama installed with `qwen2.5:3b` and `nomic-embed-text` models pulled
- Groq API key (free tier works — [console.groq.com](https://console.groq.com))

### Quick Start

```bash
# Clone
git clone https://github.com/your-username/courtRoom.ai.git
cd courtRoom.ai

# Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install

# Configure
cp .env.example .env           # Add your GROQ_API_KEY and GOOGLE_API_KEY
# (Google API key is optional — set EMBEDDING_PROVIDER=ollama to skip it)

# Run (or use start.bat)
cd ../backend
uvicorn api.main:app --reload --port 8000
# In another terminal:
cd ../frontend
npm run dev
```

Open `http://localhost:3000` and start asking legal questions.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ | — | Groq Cloud API key for LLM generation and translation |
| `GROQ_ENABLED` | No | `1` | Set to `0` to disable Groq (falls back to local Ollama) |
| `GROQ_TRANSLATION_ENABLED` | No | `1` | Set to `0` to disable Groq translation (uses Google/Ollama fallback) |
| `GROQ_GENERATION_MODEL` | No | `llama-3.3-70b-versatile` | Groq model for legal analysis |
| `GROQ_TRANSLATE_MODEL` | No | `llama-3.1-8b-instant` | Groq model for translation |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_GENERATION_MODEL` | No | `qwen2.5:3b` | Local Ollama model for fallback generation |
| `OLLAMA_EMBED_MODEL` | No | `nomic-embed-text` | Ollama model for embeddings |
| `EMBEDDING_PROVIDER` | No | `google` | Set to `ollama` to use local embeddings (no Google API key needed) |
| `GOOGLE_API_KEY` | No | — | Google AI API key for Gemini embeddings |
| `MONGODB_URI` | No | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB` | No | `courtroom_ai` | MongoDB database name |

---

## 🔧 Tech Stack

### Backend

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI 0.104 |
| LLM (cloud) | Groq SDK — `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` |
| LLM (local) | Ollama — `qwen2.5:3b` |
| Vector DB | ChromaDB 0.4 |
| Document DB | MongoDB 4.6 (via PyMongo) |
| Embeddings | Google Gemini `gemini-embedding-001` or Ollama `nomic-embed-text` |
| PDF generation | ReportLab 4.0 |
| PDF parsing | PyMuPDF 1.23, PyPDF 3.17 |
| Search | BM25 (rank-bm25) + vector ensemble |
| Reranker | BAAI/bge-reranker-base (optional, disabled by default) |
| Auth | JWT (python-jose) + bcrypt |
| Rate limiting | SlowAPI |

### Frontend

| Component | Technology |
|-----------|-----------|
| Framework | React 19.2 |
| Language | TypeScript 6.0 |
| Bundler | Vite 8.1 |
| Styling | Tailwind CSS 4.3 |
| State | Zustand 5.0 |
| Routing | React Router DOM 7.18 |
| HTTP | Axios + native fetch (SSE streaming) |
| Icons | Lucide React |

---

## 📊 Retrieval Performance

The ensemble retrieval is tuned for legal text precision:

- **BM25 weight: 0.7** — favored over vector search because section-number queries and exact legal terminology benefit from lexical matching
- **Vector weight: 0.3** — captures semantic similarity for paraphrased queries
- **Dynamic weights** — auto-switches to higher BM25 weight when section-number patterns are detected
- **Optional reranker** — BGE cross-encoder for when maximum precision is needed

---

## 🧪 Testing

Test scripts live in `C:\Users\...\opencode\` (run via temp files to avoid PowerShell encoding issues):

```bash
# Run all agent tests (33 checks — notice, RTI, strategy, audit, config)
python test_agents_all.py

# Run multi-turn continuation tests (30 checks)
python test_multiturn.py

# Run Phase 6 regression tests (3 checks)
python test_phase6.py

# Live server test against running backend on port 8000
python test_live_user.py
```

---

## 📄 License

This project is for educational and research purposes. Legal information provided by this system is not a substitute for professional legal advice.

---

> Built with ❤️ for making Indian legal information accessible to everyone.
