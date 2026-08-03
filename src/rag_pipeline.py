import os
import sys
import io
import re
import json
import sqlite3
import warnings
warnings.filterwarnings("ignore")

os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["POSTHOG_DISABLED"] = "true"

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from typing import List, Dict, Tuple, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings
import joblib
from pathlib import Path
import requests

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAG_CONFIG

_BM25_STOPWORDS = frozenset("""
a an and are as at be but by for from has have he her his i if in into is it its
me my no not of on or our ours shall she so that the their theirs them then there
these they this those to too us we what when where which who whom why with you your
yours also any being both did does doing do each few had how just more most other
some such than then they things through under until very was were will would
act acts section sections subsection sub clause chapter schedule law laws india
indian person persons people shall deemed aforesaid hereinafter thereof therein
thereto thereby thereafter thereupon said respect respects relation relating
manner provisions provision purposes purpose effect meaning expressions expression
context otherwise requires required case cases offence offences punishment
punishable provided whether further subject notwithstanding without within above
below under unless until upon per every each either neither nor not only as well
""".split())

def _bm25_tokenizer(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in words if w not in _BM25_STOPWORDS]


class _OllamaEmbedder:
    """Batch embedder using Ollama's /api/embed endpoint"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.batch_size = 50

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import json as _json, urllib.request as _req
        texts = ["search_document: " + t for t in texts]
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            body = _json.dumps({"model": self.model, "input": batch}).encode()
            req = _req.Request(
                f"{self.base_url}/api/embed",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with _req.urlopen(req, timeout=120) as resp:
                data = _json.loads(resp.read())
            results.extend(data["embeddings"])
            print(f"  [Embed] Batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1} ({len(batch)} docs)")
        return results

    def embed_query(self, text: str) -> List[float]:
        import json as _json, urllib.request as _req
        body = _json.dumps({"model": self.model, "input": ["search_query: " + text]}).encode()
        req = _req.Request(
            f"{self.base_url}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with _req.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read())
        return data["embeddings"][0]


class RAGPipeline:
    """Enhanced RAG with reranker, MMR, HyDE, and dynamic weighting"""

    def __init__(self,
                 pdf_directory: str = None,
                 chroma_db_path: str = None,
                 ollama_base_url: str = None):
        from dotenv import load_dotenv
        load_dotenv(override=True)

        self.pdf_directory = pdf_directory or os.getenv("PDF_DIRECTORY", "data/pdfs")
        self.chroma_db_path = chroma_db_path or os.getenv("CHROMA_DB_PATH", "chroma_db")
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        os.environ["CHROMA_TELEMETRY"] = "false"
        os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"

        self.embeddings = _OllamaEmbedder(self.ollama_base_url)

        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self._reranker = None
        self._all_docs_cache = None

        self.reranker_enabled = RAG_CONFIG["reranker_enabled"]
        self.hyde_enabled = RAG_CONFIG["hyde_enabled"]
        self.mmr_enabled = RAG_CONFIG["mmr_enabled"]
        self.dynamic_weights_enabled = RAG_CONFIG["dynamic_weights_enabled"]

        if os.path.exists(self.chroma_db_path) and os.listdir(self.chroma_db_path):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.chroma_db_path,
                    embedding_function=self.embeddings,
                    client_settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                chunks_cache = Path("data/chunks/bm25_docs.pkl")
                if chunks_cache.exists():
                    self._all_docs_cache = joblib.load(chunks_cache)
                    self.bm25_retriever = BM25Retriever.from_documents(
                        self._all_docs_cache,
                        preprocess_func=_bm25_tokenizer,
                        k=RAG_CONFIG["ensemble_top_k"]
                    )
                    self._build_ensemble("")
                    print("[OK] Loaded existing ChromaDB and BM25 retrievers from cache")
            except Exception as e:
                print(f"[WARN] Failed to load existing retrievers: {e}")

    # ── Reranker ──────────────────────────────────────────────────

    def _get_reranker(self):
        """Lazy-load cross-encoder reranker"""
        if self._reranker is None and self.reranker_enabled:
            try:
                os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / ".hf_cache"))
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(
                    RAG_CONFIG["reranker_model"],
                    device=RAG_CONFIG["reranker_device"]
                )
                print(f"[OK] Loaded reranker: {RAG_CONFIG['reranker_model']}")
            except Exception as e:
                print(f"[WARN] Reranker failed to load: {e}. Running without reranker.")
                self.reranker_enabled = False
        return self._reranker

    def _rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """Rerank documents using cross-encoder scores"""
        reranker = self._get_reranker()
        if not reranker:
            return docs[:top_k]

        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, s in scored[:top_k]]

    # ── Dynamic Weights ──────────────────────────────────────────

    def _has_section_ref(self, query: str) -> bool:
        """Detect if query references specific legal sections"""
        return bool(re.search(
            r'\b(?:section|article|rule|IPC|BNS|Section|Article|Rule)\s*\d+',
            query
        ))

    def _get_weights(self, query: str) -> List[float]:
        """Choose ensemble weights based on query type"""
        if not self.dynamic_weights_enabled:
            return [RAG_CONFIG["vector_weight"], RAG_CONFIG["bm25_weight"]]
        if self._has_section_ref(query):
            return RAG_CONFIG["ensemble_weights_bm25_favored"]
        return RAG_CONFIG["ensemble_weights_default"]

    def _get_chroma_kwargs(self, top_k: int) -> dict:
        """Build ChromaDB retriever kwargs based on config"""
        base = {"k": top_k}
        if self.mmr_enabled:
            base.update({
                "fetch_k": RAG_CONFIG["mmr_fetch_k"],
                "lambda_mult": RAG_CONFIG["mmr_lambda_mult"]
            })
        return base

    def _build_ensemble(self, query: str, top_k: int = 5):
        """(Re)build ensemble retriever with current settings"""
        if not self.vectorstore or not self.bm25_retriever:
            return

        chroma_kwargs = self._get_chroma_kwargs(top_k)
        weights = self._get_weights(query)
        search_type = "mmr" if self.mmr_enabled else "similarity"

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vectorstore.as_retriever(
                    search_type=search_type,
                    search_kwargs=chroma_kwargs
                ),
                self.bm25_retriever
            ],
            weights=weights
        )

    # ── HyDE ─────────────────────────────────────────────────────

    def _generate_hypothetical_doc(self, query: str) -> str:
        """Generate a hypothetical legal scenario to improve retrieval"""
        if not self.hyde_enabled:
            return query

        prompt = f"""Based on this user's legal query, write a short hypothetical legal document describing the situation as if it happened. Include relevant legal keywords and domain terms.

Query: {query}

Hypothetical scenario:"""
        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": RAG_CONFIG["hyde_max_tokens"]}
                },
                timeout=RAG_CONFIG["hyde_timeout"]
            )
            if resp.status_code == 200:
                hypo = resp.json().get("response", "").strip()
                if hypo:
                    print(f"  [HyDE] Generated hypothetical doc ({len(hypo)} chars)")
                    return hypo
        except Exception as e:
            print(f"  [HyDE] Generation failed: {e}")

        return query

    # ── Chunking ─────────────────────────────────────────────────

    def chunk_pdf_with_metadata(self, pdf_path: str) -> List[Document]:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        act_name = self._map_pdf_act_name(os.path.basename(pdf_path))

        state_keywords = {
            "bombay": ["maharashtra", "mumbai"],
            "delhi": ["delhi", "new delhi"],
            "gujarat": ["gujarati", "ahmedabad"],
            "karnataka": ["bangalore", "karnataka"],
            "tamil_nadu": ["tamil", "madras"],
            "west_bengal": ["calcutta", "bengal"],
            "punjab": ["punjab", "amritsar"],
            "rajasthan": ["jaipur", "rajasthan"]
        }

        detected_state = "Pan-India"
        for state, keywords in state_keywords.items():
            if any(kw in pdf_path.lower() for kw in keywords):
                detected_state = state.replace("_", " ").title()
                break

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CONFIG["chunk_size"],
            chunk_overlap=RAG_CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = []
        full_text = "\n\n".join(doc.page_content for doc in documents)
        for section_doc in self._split_into_sections(full_text, pdf_path):
            chunks.extend(splitter.split_documents([section_doc]))

        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            section_match = chunk.metadata.get("section_number") or self._extract_section_number(chunk.page_content)
            section_title = chunk.metadata.get("section_title") or self._extract_section_title(chunk.page_content)
            topic = self._detect_topic(section_title or chunk.page_content, act_name)
            related_sections = self._find_related_sections(chunk.page_content)
            courts = self._determine_applicable_courts(act_name, topic)
            concepts = self._extract_key_concepts(chunk.page_content)

            metadata = {
                "source": pdf_path,
                "act_name": act_name,
                "state": detected_state,
                "section_number": section_match or "",
                "section_title": section_title,
                "topic": topic,
                "related_sections": ", ".join(related_sections),
                "applicable_courts": ", ".join(courts),
                "keywords": ", ".join(concepts),
                "page_number": chunk.metadata.get("page", i),
                "chunk_index": i,
                "case_scenarios": ", ".join(self._generate_case_scenarios(topic)),
                "penalty_or_relief": self._extract_penalty_or_relief(chunk.page_content)
            }
            chunk.metadata.update(metadata)
            chunk.metadata = {k: (v if v is not None else "") for k, v in chunk.metadata.items()}
            enhanced_chunks.append(chunk)

        return enhanced_chunks

    def _split_into_sections(self, text: str, pdf_path: str) -> List[Document]:
        """Split a PDF's full text into section blocks at '303. Theft.'-style headers.

        Body sections carry an en/em dash after the title ("303. Theft .—(1) ...");
        TOC lines never do. The first dash-style header marks the start of the
        body, so everything before it (title page, TOC) is split separately and
        only kept if it is substantive (>= 200 chars). Within the body, tiny
        chunks (TOC leftovers, footnotes) are dropped.
        """
        act_name = self._map_pdf_act_name(os.path.basename(pdf_path))

        header_re = re.compile(
            r"^\s*(?:\d+\[)?(\d{1,3}[A-Z]?)\s*\.\s*[\u2013\u2014]?\s*(?:\(\d+\)\s*)?([A-Z][^\n]{2,200}?)(?:[\u2013\u2014]|$)",
            re.MULTILINE,
        )
        body_re = re.compile(
            r"^\s*(\d{1,3}[A-Z]?)\s*\.\s+([A-Z][^\u2013\u2014\n]*)[^\S\n]*[.]?[^\S\n]*[\u2013\u2014]",
            re.MULTILINE,
        )

        def build(region: str) -> List[Document]:
            matches = list(header_re.finditer(region))
            if not matches:
                return [Document(page_content=region, metadata={"section_number": "", "section_title": "", "source": pdf_path, "act_name": act_name})]
            sections = []
            current_num, current_title, current_start = "", "", 0
            for i, m in enumerate(matches):
                if i == 0:
                    current_num, current_title = m.group(1), self._clean_section_title(m.group(2))
                    current_start = m.end()
                    continue
                sections.append(Document(
                    page_content=region[current_start:m.start()].strip(),
                    metadata={"section_number": f"Section {current_num}", "section_title": current_title,
                              "source": pdf_path, "act_name": act_name}
                ))
                current_num, current_title = m.group(1), self._clean_section_title(m.group(2))
                current_start = m.end()
            if matches:
                sections.append(Document(
                    page_content=region[current_start:].strip(),
                    metadata={"section_number": f"Section {current_num}", "section_title": current_title,
                              "source": pdf_path, "act_name": act_name}
                ))
            return sections

        body_start = body_re.search(text)
        if body_start:
            pre = build(text[:body_start.start()])
            body = build(text[body_start.start():])
            return [s for s in pre if len(s.page_content) >= 200 and not self._is_amendment_note(s)] + \
                   [s for s in body if len(s.page_content) >= 60 and not self._is_amendment_note(s)]
        return [s for s in build(text) if len(s.page_content) >= 60 and not self._is_amendment_note(s)]

    @staticmethod
    def _is_amendment_note(doc: Document) -> bool:
        """Drop footnote/amendment-note chunks like 'Subs. by Act 41 of 2005, s. 8...'"""
        title = (doc.metadata.get("section_title") or "").strip().lower()
        return bool(re.match(r"^(subs\.|omitted by|ins\. by|added by|substituted by|the words)", title))

    @staticmethod
    def _clean_section_title(raw: str) -> str:
        """'Theft .—(1) Whoever...' -> 'Theft'; 'Short title and extent.  ' -> 'Short title and extent'."""
        cut = re.search(r"[\u2013\u2014]", raw)
        if cut:
            raw = raw[:cut.start()]
        return raw.strip(" .\t\u00a0")

    def _map_pdf_act_name(self, filename: str) -> str:
        """Map PDF filenames to clean act names for metadata"""
        act_map = {
            "BNS_2023.pdf": "Bharatiya Nyaya Sanhita 2023",
            "IPC_1860.pdf": "Indian Penal Code 1860",
            "Consumer_Protection_Act_2019.pdf": "Consumer Protection Act 2019",
            "RTI_Act_2005.pdf": "Right to Information Act 2005",
            "IT_Act_2000.pdf": "Information Technology Act 2000",
            "payment_of_wages_act_1936.pdf": "Payment of Wages Act 1936",
            "Minimum_Wages_Act_1948.pdf": "Minimum Wages Act 1948",
            "Industrial_Disputes_Act_1947.pdf": "Industrial Disputes Act 1947",
            "Gujarat_Rent_Control_Act_1999.pdf": "Bombay Rents, Hotel and Lodging House Rates Control Act 1947 (Gujarat)",
            "Bombay Rents_Hotel and Lodging House Rates Control Act_1947.pdf": "Bombay Rents Hotel and Lodging House Rates Control Act 1947",
            "IPC Section @lawforcivilservices.pdf": "Indian Penal Code 1860 (Section Notes)",
            "CRPC Sec One Liner.pdf": "Code of Criminal Procedure 1973 (Section Notes)",
            "CPC With Chart @lawforcivilservices.pdf": "Code of Civil Procedure 1908 (Chart Notes)",
            "Constitutional-Amendment.pdf": "Constitution of India 1950 (Amendments)",
            "Constitutional-Bodies-in-India.pdf": "Constitution of India 1950 (Constitutional Bodies)",
            "Constution Amendment @lawforcivilservices.pdf": "Constitution of India 1950 (Amendments)",
            "Fundamental-rights.pdf": "Constitution of India 1950 (Fundamental Rights)",
            "Tax Guide.pdf": "Tax Laws of India (Guide)",
            "Triple Talaq @lawforcivilservices.pdf": "Muslim Law (Triple Talaq)",
        }
        return act_map.get(filename, filename.replace(".pdf", "").replace("_", " "))

    # ── Metadata helpers ─────────────────────────────────────────

    def _extract_section_number(self, text: str) -> Optional[str]:
        patterns = [
            r"Section\s+(\d+[A-Z]*)",
            r"Article\s+(\d+)",
            r"Rule\s+(\d+)",
            r"Schedule\s+(\d+)",
            r"^\s*(\d{1,3}[A-Z]?)\s*\.\s+[A-Z]",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                num = match.group(1)
                if pattern.startswith(r"^\s*"):
                    return f"Section {num}"
                return match.group(0)
        return None

    def _extract_section_title(self, text: str) -> str:
        """Extract section title from header line like '303. Theft .—(1) Whoever...'"""
        match = re.search(r"^\s*\d{1,3}[A-Z]?\s*\.\s+([A-Z][A-Za-z()& ,\-']{2,60})", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return ""

    def _detect_topic(self, text: str, act_name: str) -> str:
        topic_keywords = {
            "Criminal": ["intimidation", "harassment", "threat", "assault", "offence", "offense", "crime", "criminal", "theft", "robbery", "murder", "kidnapping", "cheating", "fraud", "dacoity", "extortion", "penal"],
            "Civil": ["damages", "injunction", "relief", "compensation", "lawsuit", "suit", "decree", "plaint", "civil"],
            "Eviction": ["eviction", "tenancy", "landlord", "tenant", "rent"],
            "Payment": ["wages", "salary", "payment", "refund"],
            "Property": ["property", "land", "building", "premises", "possession", "ownership", "immovable"],
            "Family": ["marriage", "divorce", "custody", "maintenance", "inheritance", "talaq", "dowry"],
            "Labor": ["employment", "worker", "employee", "wages", "industrial dispute", "labour"],
            "Contract": ["agreement", "contract", "breach of trust", "negotiable instrument"],
            "Cyber": ["online", "digital", "internet", "email", "data", "cyber", "electronic"],
            "Consumer": ["consumer", "goods", "service", "defective", "refund", "spurious"],
            "Tax": ["tax", "income", "gst", "assessment", "deduction"],
            "Constitutional": ["constitution", "fundamental right", "directive principle", "amendment", "article", "writ"],
            "Evidence": ["evidence", "witness", "presumption", "proof", "admissible"],
        }
        text_lower = text.lower()
        scores = {topic: 0 for topic in topic_keywords}
        for topic, keywords in topic_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[topic] += 1
        best_topic = max(scores, key=lambda t: scores[t])
        if scores[best_topic] > 0:
            return best_topic
        if any(kw in act_name.lower() for kw in ["criminal", "penal", "bns", "ipc"]):
            return "Criminal"
        return "General Legal"

    def _find_related_sections(self, text: str) -> List[str]:
        pattern = r"(?:refer to |see |under |pursuant to |as per ).*?(?:Section|Article|Rule)\s+(\d+[A-Z]*)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(set(matches))[:5]

    def _determine_applicable_courts(self, act_name: str, topic: str) -> List[str]:
        courts = {"District Court"}
        if "civil" in topic.lower():
            courts.add("Civil Court")
        if "criminal" in topic.lower():
            courts.add("Sessions Court")
            courts.add("Magistrate Court")
        if "eviction" in topic.lower() or "rent" in act_name.lower():
            courts.add("Rent Court")
        if "consumer" in topic.lower() or "consumer" in act_name.lower():
            courts.add("Consumer Court")
        if "labour" in topic.lower() or "labour" in act_name.lower():
            courts.add("Labour Court")
            courts.add("Industrial Tribunal")
        if "family" in act_name.lower():
            courts.add("Family Court")
        if "cyber" in topic.lower() or "cyber" in act_name.lower():
            courts.add("Cyber Crime Cell")
        return list(courts)

    def _extract_key_concepts(self, text: str) -> List[str]:
        concepts = []
        legal_terms = {
            "illegal": ["illegal", "unlawful", "prohibited"],
            "breach": ["breach", "violation", "infringement"],
            "negligence": ["negligence", "carelessness", "failure"],
            "tort": ["tort", "wrongful"],
            "damages": ["damages", "compensation", "relief"],
            "injunction": ["injunction", "restraining order"],
            "specific_performance": ["specific performance"],
            "possession": ["possession", "occupation", "residence"],
            "harassment": ["harassment", "intimidation", "threats"],
            "defamation": ["defame", "slander", "libel"],
            "extortion": ["extortion", "blackmail", "coercion"]
        }
        text_lower = text.lower()
        for concept, keywords in legal_terms.items():
            if any(kw in text_lower for kw in keywords):
                concepts.append(concept)
        return concepts

    def _generate_case_scenarios(self, topic: str) -> List[str]:
        scenarios = {
            "Criminal": ["Intimidation and threats", "Harassment and abuse", "Wrongful restraint", "Breach of peace", "Assault"],
            "Civil": ["Contract breach", "Damages recovery", "Injunction relief", "Specific performance", "Property disputes"],
            "Eviction": ["Illegal eviction attempt", "Wrongful lock-out", "Utility disconnection", "Non-payment of rent", "Breach of tenancy"],
            "Labor": ["Wrongful termination", "Non-payment of wages", "Unsafe working conditions", "Discrimination", "Sexual harassment"]
        }
        return scenarios.get(topic, ["General legal matter"])

    def _extract_penalty_or_relief(self, text: str) -> Optional[str]:
        patterns = [
            r"punishment.*?(?:imprisonment|fine|both)",
            r"penalty.*?(?:rupees?|fine)",
            r"fine.*?(?:rupees?|\d+)",
            r"imprisonment.*?(?:years?|months?)",
            r"compensation.*?(?:rupees?|amount)"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0)[:100]
        return None

    # ── JSON Laws Loader ────────────────────────────────────────

    def _load_json_laws(self) -> List[Document]:
        """Load all structured JSON law files from data/laws/ into Documents"""
        laws_dir = Path("data/laws")
        if not laws_dir.exists():
            print("  [JSON] No data/laws/ directory found")
            return []

        documents = []
        act_name_map = {
            "ipc": "Indian Penal Code 1860 (mapped to BNS 2023)",
            "crpc": "Code of Criminal Procedure 1973 (mapped to BNSS 2023)",
            "cpc": "Code of Civil Procedure 1908",
            "coi": "Constitution of India 1950",
            "iea": "Indian Evidence Act 1872 (mapped to BSA 2023)",
            "hma": "Hindu Marriage Act 1955",
            "ida": "Industrial Disputes Act 1947",
            "mva": "Motor Vehicles Act 1988",
            "nia": "Negotiable Instruments Act 1881",
        }

        for json_path in sorted(laws_dir.glob("*.json")):
            stem = json_path.stem.lower()
            if stem != "coi":
                continue
            act_name = act_name_map.get(stem, stem.replace("_", " ").title())
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [JSON] Failed to parse {json_path.name}: {e}")
                continue

            entries = []
            if isinstance(raw, list):
                if raw and isinstance(raw[0], list):
                    for inner in raw:
                        if isinstance(inner, list):
                            entries.extend(inner)
                        else:
                            entries.append(inner)
                else:
                    entries = raw

            count = 0
            for item in entries:
                if not isinstance(item, dict):
                    continue

                doc = self._json_item_to_doc(item, act_name, json_path.name)
                if doc:
                    documents.append(doc)
                    count += 1

            print(f"  [JSON] {json_path.name}: {count} sections loaded")

        print(f"  [JSON] Total: {len(documents)} sections from all law files")
        return documents

    def _load_db_laws(self) -> List[Document]:
        """Load structured law sections from IndiaLaw.db (clean canonical source)"""
        db_path = Path("data/laws") / "IndiaLaw.db"
        if not db_path.exists():
            print(f"  [DB] No IndiaLaw.db found at {db_path}")
            return []

        act_name_map = {
            "IPC": "Indian Penal Code 1860 (mapped to BNS 2023)",
            "CRPC": "Code of Criminal Procedure 1973 (mapped to BNSS 2023)",
            "CPC": "Code of Civil Procedure 1908",
            "IEA": "Indian Evidence Act 1872 (mapped to BSA 2023)",
            "HMA": "Hindu Marriage Act 1955",
            "IDA": "Indian Divorce Act 1869",
            "MVA": "Motor Vehicles Act 1988",
            "NIA": "Negotiable Instruments Act 1881",
        }

        documents = []
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for table in sorted(tables):
                if table not in act_name_map:
                    print(f"  [DB] Skipping unknown table: {table}")
                    continue
                act_name = act_name_map[table]
                cols = [c[1] for c in cur.execute(f'PRAGMA table_info("{table}")')]
                sec_col = "Section" if "Section" in cols else "section"
                desc_col = "section_desc" if "section_desc" in cols else "description"
                title_col = "section_title" if "section_title" in cols else "title"
                chapter_col = "chapter" if "chapter" in cols else None

                count = 0
                for row in cur.execute(f'SELECT * FROM "{table}"'):
                    item = dict(zip(cols, row))
                    section_desc = (item.get(desc_col) or "").strip()
                    if not section_desc:
                        continue
                    section_num = item.get(sec_col, "")
                    section_title = (item.get(title_col) or "").strip()
                    chapter = item.get(chapter_col, "") if chapter_col else ""
                    topic = self._detect_topic(section_desc + " " + section_title, act_name)

                    documents.append(Document(
                        page_content=section_desc,
                        metadata={
                            "source": f"IndiaLaw.db::{table}",
                            "act_name": act_name,
                            "section_number": f"Section {section_num}" if str(section_num).strip() else "",
                            "section_title": section_title,
                            "chapter": chapter,
                            "topic": topic,
                            "state": "Pan-India",
                            "applicable_courts": "District Court",
                            "source_act": act_name,
                            "keywords": ", ".join(self._extract_key_concepts(section_desc)),
                        }
                    ))
                    count += 1
                print(f"  [DB] {table}: {count} sections loaded")
        finally:
            conn.close()

        print(f"  [DB] Total: {len(documents)} sections from IndiaLaw.db")
        return documents

    def _json_item_to_doc(self, item: dict, act_name: str, source_file: str) -> Optional[Document]:
        """Convert a single JSON law entry to a Document, handling all formats"""
        keys = list(item.keys())

        # Broken CSV-key format (hma.json): {"chapter,section,section_title,section_desc": "1,1,Title,\"desc\""}
        if len(keys) == 1 and "," in keys[0]:
            val = item[keys[0]]
            if not val or not val.strip():
                return None
            parts = self._parse_csv_line(val)
            section_num = parts[0] if len(parts) > 0 else ""
            section_title = parts[1] if len(parts) > 1 else ""
            section_desc = parts[2] if len(parts) > 2 else ""

            text = section_desc or section_title
            if not text:
                return None
            return Document(
                page_content=text,
                metadata={
                    "source": source_file,
                    "act_name": act_name,
                    "section_number": f"Section {section_num}" if section_num else "",
                    "section_title": section_title,
                    "topic": "Family Law" if "marriage" in act_name.lower() else "General",
                    "state": "Pan-India",
                    "applicable_courts": "District Court",
                    "source_act": act_name,
                }
            )

        # Constitution format: {"ArtNo", "Name", "ArtDesc"}
        if "ArtNo" in item:
            section_desc = item.get("ArtDesc", "")
            if not section_desc:
                return None
            return Document(
                page_content=section_desc,
                metadata={
                    "source": source_file,
                    "act_name": act_name,
                    "section_number": f"Article {item['ArtNo']}",
                    "section_title": item.get("Name", ""),
                    "topic": "Constitutional Law",
                    "state": "Pan-India",
                    "applicable_courts": "Supreme Court, High Court",
                    "source_act": act_name,
                }
            )

        # Full format: {"chapter", "section", "section_title", "section_desc"}
        section_desc = item.get("section_desc") or item.get("description", "")
        if not section_desc:
            return None

        section_num = item.get("Section") or item.get("section", "")
        section_title = item.get("section_title") or item.get("title", "")
        chapter = item.get("chapter", "")

        section_label = f"Section {section_num}" if section_num else ""
        topic = self._detect_topic(section_desc + " " + section_title, act_name)

        return Document(
            page_content=section_desc,
            metadata={
                "source": source_file,
                "act_name": act_name,
                "section_number": section_label,
                "section_title": section_title,
                "chapter": chapter,
                "topic": topic,
                "state": "Pan-India",
                "applicable_courts": "District Court",
                "source_act": act_name,
                "keywords": ", ".join(self._extract_key_concepts(section_desc)),
            }
        )

    def _parse_csv_line(self, line: str) -> List[str]:
        """Parse a CSV line that may contain quoted fields with commas"""
        parts = []
        current = ""
        in_quotes = False
        for ch in line:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == "," and not in_quotes:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        parts.append(current.strip())
        return parts

    def build_knowledge_base(self):
        print("🔄 Building knowledge base...")

        all_chunks = []
        os.makedirs(self.pdf_directory, exist_ok=True)
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_directory, filename)
                print(f"  📄 Processing: {filename}")
                try:
                    chunks = self.chunk_pdf_with_metadata(pdf_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"  ⚠️  Error processing {filename}: {e}")

        print(f"✅ Total chunks created: {len(all_chunks)}")

        # Load structured laws from IndiaLaw.db + COI.json
        print("🔄 Loading structured laws from IndiaLaw.db...")
        db_docs = self._load_db_laws()
        print("🔄 Loading structured JSON laws (COI)...")
        json_docs = self._load_json_laws()
        all_docs = all_chunks + db_docs + json_docs

        if not all_docs:
            print("[WARN] No documents created. Add PDFs to data/pdfs or JSON files to data/laws.")
            return []

        chunks_cache_dir = Path("data/chunks")
        chunks_cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(all_docs, chunks_cache_dir / "bm25_docs.pkl")

        print("🔄 Creating vector store...")
        if self.vectorstore is not None:
            try:
                self.vectorstore._client.delete_collection(self.vectorstore._collection.name)
                print(f"  [Chroma] Dropped existing collection: {self.vectorstore._collection.name}")
            except Exception as e:
                print(f"  [Chroma] No existing collection to drop: {e}")
        self.vectorstore = None
        batch_size = 100
        first_batch = True
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i+batch_size]
            for attempt in range(3):
                try:
                    if first_batch:
                        self.vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory=self.chroma_db_path,
                            client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
                        )
                        first_batch = False
                    else:
                        self.vectorstore.add_documents(batch)
                    print(f"  [Chroma] Batch {i//batch_size + 1}/{(len(all_docs)-1)//batch_size + 1} ({len(batch)} docs)")
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  [Chroma] Batch {i//batch_size + 1} failed, retry {attempt+1}: {e}")
                        import time; time.sleep(3)
                    else:
                        raise
        self.vectorstore.persist()
        print("✅ Vector store created and persisted")

        print("🔄 Creating BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(
            all_docs,
            preprocess_func=_bm25_tokenizer,
            k=RAG_CONFIG["ensemble_top_k"]
        )
        print("✅ BM25 retriever created")

        self._build_ensemble("", top_k=RAG_CONFIG["top_k_retrieval"])
        print("✅ Ensemble retriever created")

        return all_docs

    # ── Retrieval ────────────────────────────────────────────────

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
        "mob lynching": "lynching", "lynched": "lynching",
        "domestic violence": "domestic violence", "dowry": "dowry",
        "gst": "gst", "tax": "tax", "income tax": "income tax",
        "bounced": "dishonour of cheque returned by the bank unpaid", "bounce": "dishonour of cheque returned by the bank unpaid",
        "bouncing": "dishonour of cheque returned by the bank unpaid", "dishonoured": "dishonour of cheque returned by the bank unpaid",
        "fired": "retrenchment", "sacked": "retrenchment", "laid off": "retrenchment", "laid-off": "retrenchment",
        "terminated": "retrenchment", "fired me": "retrenchment",
        "hit by bike": "road accident", "hit by car": "road accident", "hit by a bike": "road accident",
        "hit by a car": "road accident", "hit by a vehicle": "road accident", "hit by truck": "road accident",
        "bike hit": "road accident", "accident claim": "accident claim compensation",
        "refuses to give me divorce": "decree of divorce", "give me divorce": "decree of divorce",
        "want a divorce": "decree of divorce", "want divorce": "decree of divorce",
        "get a divorce": "decree of divorce",
        "threatened": "criminal intimidation", "threaten": "criminal intimidation",
        "threatening": "criminal intimidation", "threats": "criminal intimidation",
        "threat": "criminal intimidation",
        "hacked": "unauthorised access", "hacking": "unauthorised access", "hack": "unauthorised access",
        "broke into": "house-trespass",
        "sold my flat to someone else": "cheating", "sold my flat": "cheating", "sold the flat": "cheating",
        "sold our flat": "cheating", "sold the flat to someone else": "cheating",
        "won't return": "criminal breach of trust", "will not return": "criminal breach of trust",
        "refused to return": "criminal breach of trust", "refuses to return": "criminal breach of trust",
        "won't give back": "criminal breach of trust", "refusing to return": "criminal breach of trust",
        "asking for more money": "dowry demand", "asked for more money": "dowry demand",
        "asking for money": "dowry demand", "asking for dowry": "dowry demand", "demanded dowry": "dowry demand",
        "slapped": "cruelty", "slap": "cruelty", "slaps": "cruelty", "hit me": "cruelty",
        "beat me": "cruelty", "beaten": "cruelty", "beating me": "cruelty", "beats me": "cruelty",
        "assaulted": "cruelty", "assault": "cruelty", "harassed": "cruelty", "harassing": "cruelty",
        "harassment": "cruelty",
        "threw me out": "cruelty", "threw me": "cruelty", "kicked me out": "cruelty",
        "thrown me out": "cruelty", "thrown out of": "cruelty", "turned me out": "cruelty",
        "stopped working": "defect in goods", "not working": "defect in goods", "broke down": "defect in goods",
        "stopped functioning": "defect in goods", "stopped working on its own": "defect in goods",
        "snatched": "snatching", "snatch": "snatching", "snatcher": "snatching", "snatchers": "snatching",
        "refuses to give back": "criminal breach of trust", "refuse to give back": "criminal breach of trust",
        "not giving back": "criminal breach of trust", "not returning": "criminal breach of trust",
        "didn't give back": "criminal breach of trust", "didn't return": "criminal breach of trust",
        "divorced": "divorce", "divorcing": "divorce", "divorcee": "divorce",
        "salary": "wages", "salaries": "wages", "my salary": "my wages",
        "has not paid my salary": "time of payment of wages", "not paying my salary": "time of payment of wages",
        "salary not paid": "time of payment of wages", "did not pay my salary": "time of payment of wages",
        "hasn't paid my salary": "time of payment of wages", "not paid my wages": "time of payment of wages",
        "fired": "retrenchment", "sacked": "retrenchment", "laid off": "retrenchment", "laid-off": "retrenchment",
        "terminated": "retrenchment", "retrenched": "retrenchment",
        "rti": "right to information", "rti application": "right to information application",
    }

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

    def _title_boost(self, doc, legal_terms: set, query: str = "") -> int:
        """Score how strongly the section title matches query legal terms.
        Title-phrase present in query (criminal intimidation) beats exact
        single-word titles (definitional sections like 'Cheque'), which beat
        single-term word-boundary matches."""
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

    def _score_pool(self, query: str) -> dict:
        """Ensemble-equivalent combined relevance scores for the corpus,
        keyed by content hash: 0.3 * vector relevance + 0.7 * BM25 (normalized).
        The EnsembleRetriever itself discards these scores (and its documents
        are object copies), so we recompute them keyed by content to break
        title-boost ties deterministically."""
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

    def _normalize_query(self, query: str) -> str:
        """Map common user phrasing to legal terms for better BM25 retrieval"""
        q = query.lower()
        for phrase, replacement in self._QUERY_SYNONYMS.items():
            q = re.sub(r"\b" + re.escape(phrase) + r"\b", replacement, q)
        return q

    def retrieve_with_metadata(self, query: str, top_k: int = None) -> List[Dict]:
        if not self.ensemble_retriever:
            raise Exception("Knowledge base not built. Call build_knowledge_base() first.")

        top_k = top_k or RAG_CONFIG["top_k_retrieval"]
        ensemble_top_k = RAG_CONFIG["ensemble_top_k"]

        # Normalize user phrasing to legal terms (helps BM25 match statute text)
        retrieval_query = self._normalize_query(query)

        # HyDE: generate hypothetical doc for better dense retrieval
        dense_query = self._generate_hypothetical_doc(retrieval_query) if self.hyde_enabled else retrieval_query

        # Rebuild ensemble with dynamically chosen weights + MMR settings
        self._build_ensemble(retrieval_query, top_k=ensemble_top_k)

        # Retrieve
        docs = self.ensemble_retriever.get_relevant_documents(dense_query)

        # Rerank
        if self.reranker_enabled and len(docs) > top_k:
            docs = self._rerank(query, docs, top_k)
        else:
            # Title boost on the FULL ensemble pool (fixes BM25 term-frequency
            # bias: theft-definition chunks rank below explanation examples)
            legal_terms = set(
                w for w in re.findall(r"[a-z]{4,}", retrieval_query)
                if w not in self._TITLE_BOOST_STOPWORDS
            )
            if legal_terms:
                # Title-match merge: pull in any section whose title matches a
                # query legal term/phrase even if BM25/vector missed it (long
                # mixed queries spread BM25 votes across many generic terms and
                # can starve distinctive sections like Criminal intimidation)
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

        # Dedupe by (act, section) — keep the best chunk per section so the
        # top-k covers distinct provisions instead of one section's fragments.
        # Drop mapped-IPC copies of a section title that already appears
        # (BNS 351 > IPC 503, BNS 316 > IPC 405...).
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

        results = []
        for i, doc in enumerate(docs):
            result = {
                "index": i,
                "content": doc.page_content,
                "text": doc.page_content,
                "act_name": doc.metadata.get("act_name", "Unknown"),
                "section_number": doc.metadata.get("section_number", "Unknown"),
                "section_title": doc.metadata.get("section_title", ""),
                "state": doc.metadata.get("state", "Pan-India"),
                "topic": doc.metadata.get("topic", "General"),
                "applicable_courts": self._split_field(doc.metadata.get("applicable_courts")),
                "keywords": self._split_field(doc.metadata.get("keywords")),
                "related_sections": self._split_field(doc.metadata.get("related_sections")),
                "case_scenarios": self._split_field(doc.metadata.get("case_scenarios")),
                "penalty_or_relief": doc.metadata.get("penalty_or_relief", None),
                "page_number": doc.metadata.get("page_number", 0),
                "source_act": doc.metadata.get("act_name", "Unknown"),
                "courts": self._split_field(doc.metadata.get("applicable_courts")),
                "metadata": doc.metadata
            }
            results.append(result)

        return results

    @staticmethod
    def _split_field(val) -> list:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return []

    def retrieve_dense(self, query: str, domain: str, top_k: int = 4) -> List[Dict]:
        if not self.ensemble_retriever:
            self.__init__(chroma_db_path=self.chroma_db_path, ollama_base_url=self.ollama_base_url)
        try:
            return self.retrieve_with_metadata(query, top_k=top_k)
        except Exception:
            return []

    def search_by_jurisdiction(self, query: str, state: str, top_k: int = 5) -> List[Dict]:
        all_results = self.retrieve_with_metadata(query, top_k=20)
        filtered = [r for r in all_results if state.lower() in r["state"].lower() or r["state"] == "Pan-India"]
        return filtered[:top_k]

    def search_by_topic(self, query: str, topic: str, top_k: int = 5) -> List[Dict]:
        all_results = self.retrieve_with_metadata(query, top_k=20)
        filtered = [r for r in all_results if topic.lower() in r["topic"].lower()]
        return filtered[:top_k]

    def get_related_laws(self, section: str) -> List[Dict]:
        query = f"Related to {section}"
        results = self.retrieve_with_metadata(query, top_k=10)
        related = [
            r for r in results
            if section in r.get("related_sections", []) or section in r.get("keywords", [])
        ]
        return related[:5]


if __name__ == "__main__":
    rag = RAGPipeline()
    chunks = rag.build_knowledge_base()

    query = "I rented a flat but landlord is cutting water supply"
    results = rag.retrieve_with_metadata(query)

    print("\n=== RETRIEVAL RESULTS ===")
    for r in results[:3]:
        print(f"\n📄 {r['act_name']} - {r['section_number']}")
        print(f"   Topic: {r['topic']}")
        print(f"   Courts: {', '.join(r['applicable_courts'])}")
        print(f"   State: {r['state']}")
