# 🏛️ courtRoom.ai: Autonomous Indian Legal AI & Advocate Copilot

[![License](https://img.shields.io/badge/License-Apache_2.0_OR_MIT-blue.svg)](LICENSE-APACHE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Groq](https://img.shields.io/badge/Inference-Groq_LPU_Qwen_27B-F55036.svg)](https://groq.com)
[![Statutory Coverage](https://img.shields.io/badge/Statutes-45+_Central_Acts_(BNS/BNSS/BSA_2023)-2E7D32.svg)](data/laws/act_registry.json)
[![Tests](https://img.shields.io/badge/QA_Suite-20%2F20_Passed_(100%25)-brightgreen.svg)](#-verification--testing)

**courtRoom.ai** is an open-source, full-stack Indian Legal AI platform and citizen advocate copilot. It grounds conversational legal queries in operative Indian statutes, provides high-precision hybrid statutory retrieval, and autonomously drafts advocate-grade pleadings with dual **print-ready A4 PDF** and **editable Microsoft Word (.docx)** export.

Fully compliant with India's post-July 2024 criminal law reform (**Bharatiya Nyaya Sanhita 2023**, **Bharatiya Nagarik Suraksha Sanhita 2023**, **Bharatiya Sakshya Adhiniyam 2023**) and labor code consolidation (**Code on Wages 2019**).

---

## ⚡ Core Capabilities

- ⚖️ **Autonomous Legal Drafting (PDF & DOCX)**: Instantly generates high-court grade legal demand notices, RTI Form A applications, cyber fraud bank freeze letters, SHO police complaints, and labour dues claims with receiving counterfoils and summary tables.
- 🧠 **Qwen 27B on Groq LPUs**: Legal pleading prose generated in ~1.2s using `qwen/qwen3.8-27b` via Groq LPUs, with multi-model fallback cascade (`qwen-qwq-32b` → `llama-3.3-70b` → `llama-3.1-8b`) and deterministic offline templates.
- 📚 **3-Tier Grounded Statutory RAG**: Zero-hallucination legal retrieval across 45+ Central Indian Statutes using hybrid BM25 + dense vector embeddings with strict word-boundary gating (`\b`) to eliminate off-topic statute leakage.
- 🔄 **Multi-Turn Persistent Sessions**: Context-aware parameter collection in MongoDB (`courtroom_ai.tool_sessions`) with native TTL indexing. Handles multi-turn clarifications without premature session clearing.
- 🇮🇳 **11 Indian Languages**: End-to-end conversational legal advisory in English, Hindi, Gujarati, Marathi, Tamil, Telugu, Kannada, Bengali, Malayalam, Punjabi, and Urdu.
- ⏱️ **Limitation Act & Deadline Calculator**: Evaluates limitation windows under the Limitation Act 1963, CPA 2019 (2 years), Code on Wages (3 years), and RTI Appeals (30/90 days).
- 📋 **BSA 2023 Evidence Checklists**: Domain-tailored evidentiary checklists compliant with Section 63 Bharatiya Sakshya Adhiniyam (electronic records certification).

---

## 🤖 The 6 Core Citizen Legal Tools

| Tool | Operative Statutory Provisions | Output Formats |
|---|---|:---:|
| 📄 **Legal Demand Notice** | Section 106/108 Transfer of Property Act 1882, Section 73 Indian Contract Act 1872 | **PDF + .DOCX** |
| 🛡️ **Cyber Fraud & Bank Freeze** | RBI Zero Liability Circular (2017), Sec 43/66D IT Act 2000, Sec 318(4) BNS 2023 | **PDF + .DOCX** |
| 🚨 **Formal Police Complaint** | Section 173 BNSS 2023, Section 304 BNS 2023 (Snatching), Sec 63 BSA 2023 | **PDF + .DOCX** |
| 💼 **Labour & FnF Recovery** | Section 17(2) & 18 Code on Wages 2019 (2-day settlement), Sec 15 Payment of Wages Act 1936 | **PDF + .DOCX** |
| 📝 **RTI Application (Form A)** | Section 6(1) Right to Information Act 2005, Central RTI Rules 2012 | **PDF + .DOCX** |
| 🗺️ **Case Strategy Roadmap** | Multi-forum dispute analysis (RERA, Consumer Commission, Civil/Labour Court) | **PDF + .DOCX** |

---

## 🏗️ Technical Architecture

```
courtRoom.ai/
├── backend/                         # FastAPI 0.115 + Python 3.11+
│   ├── api/
│   │   ├── main.py                  # REST API, universal /download/{id}, multi-turn /query
│   │   └── auth.py                  # JWT authentication & session credentials
│   ├── src/
│   │   ├── agent.py                 # Tool router, Indian rupee parser, entity extraction
│   │   ├── document_factory.py      # Dual ReportLab PDF & python-docx DOCX engine
│   │   ├── groq_drafter.py          # Qwen 27B Groq LPU drafter + model cascade
│   │   ├── session_manager.py       # MongoDB TTL multi-turn session persistence
│   │   ├── full_rag.py              # Hybrid BM25 (70%) + Vector (30%) RAG pipeline
│   │   ├── applicability.py         # Word-boundary applicability gating
│   │   ├── fact_extractor.py        # Indian entity & monetary extraction (lakh/crore)
│   │   ├── response_formatter.py    # BNS/BNSS/BSA citation verification
│   │   └── agents/                  # Specialized legal domain agents
│   └── config/                      # Legal taxonomy & compensation rules
├── frontend/                        # React 19 + TypeScript + Vite 8 + Tailwind CSS 4
│   ├── src/
│   │   ├── App.tsx                  # Interactive chat, universal PDF/DOCX downloads
│   │   └── lib/api.ts               # Streaming API client
├── data/laws/                       # 45+ JSON statutes & act_registry.json
└── .env.example                     # Environment configuration template
```

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB 6.0+ (running locally or MongoDB Atlas)
- Groq Cloud API Key ([console.groq.com](https://console.groq.com))

### 1. Clone & Configure
```bash
git clone https://github.com/Vatsal728/courtRoom.ai.git
cd courtRoom.ai

# Copy environment template
copy .env.example .env
```

Edit `.env` and add your keys:
```ini
GROQ_API_KEY=gsk_your_actual_groq_key_here
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=courtroom_ai
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r backend/requirements.txt

# Run FastAPI backend
uvicorn api.main:app --app-dir backend --reload --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🧪 Verification & Testing

The platform includes an automated master verification suite testing tool routing, entity extraction, session persistence, word-boundary gating, and dual PDF/DOCX generation:

```bash
# Execute 20-test master QA suite
python backend/tests/run_full_verification_suite.py
```

```text
==================================================================================
VERIFICATION SUMMARY: 20/20 TEST CASES PASSED (100.0%)
==================================================================================
  [PASS] TC-01 | Cyber Fraud & Bank Freeze Routing + VPA/UTR Extraction
  [PASS] TC-02 | Snatching Police Complaint (Sec 304 BNS & Sec 173 BNSS)
  [PASS] TC-03 | Unpaid Salary + Gratuity Sum & Company Extraction
  [PASS] TC-04 | RTI Form A Application Routing & PIO Extraction
  [PASS] TC-05 | Single-Turn Security Deposit Legal Notice Routing
  [PASS] TC-06 | Case Strategy Builder with Lakh Multiplier & Limitation Date
  [PASS] TC-07 | Multi-Turn Legal Notice to Tenant Ramesh (Sec 106 TPA)
  [PASS] TC-08 | Multi-Turn Labour Dues Continuation with Domain Keywords
  [PASS] TC-09 | MongoDB / In-Memory Session TTL Expiration Enforcement
  [PASS] TC-10 | DomainClassifier & FactExtractor on Cyber Fraud Benchmark
  [PASS] TC-11 | Elimination of False-Positive Cruelty Trigger on 'Wife'
  [PASS] TC-12 | Word-Boundary Gate Prevents 'con' Substring Leakage
  [PASS] TC-13 | ResponseFormatter Preserves BNS/BNSS/BSA Citations
  [PASS] TC-14 | Police Complaint PDF/DOCX Parity, Counterfoil & XML Safety
  [PASS] TC-15 | Cyber Fraud PDF/DOCX Grounding (RBI Circular & IT Act)
  [PASS] TC-16 | Labour Dues PDF/DOCX Code on Wages & 7-Day Cure Window
  [PASS] TC-17 | Parameter Alias Normalization Across Direct REST Endpoints
  [PASS] TC-18 | DeadlineTracker Multi-Format Date Parsing & Unit Accuracy
  [PASS] TC-19 | EvidenceChecklist Domain Coverage ('labor' & 'labour')
  [PASS] TC-20 | DocumentAudit Deterministic Clause Verification
```

---

## 📜 License & Disclaimer

This project is open-source and dual-licensed under either:
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))
- **MIT License** ([LICENSE-MIT](LICENSE-MIT))

at your option.

> **Disclaimer**: courtRoom.ai is an artificial intelligence research platform designed to provide automated statutory guidance and pleading drafts. It does not constitute formal legal counsel and does not create an advocate-client relationship. Users should review generated drafts with a licensed advocate before filing in court or before statutory authorities.
