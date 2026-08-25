import os
import sys
import io
import re
import json
import sqlite3
import threading
import warnings
from collections import OrderedDict
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

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAG_CONFIG
from src.paths import resolve

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
    """Enhanced RAG with reranker, MMR, and dynamic weighting"""

    _DB_ACT_MAP = {
        "IPC": "Indian Penal Code 1860 (mapped to BNS 2023)",
        "CRPC": "Code of Criminal Procedure 1973 (mapped to BNSS 2023)",
        "CPC": "Code of Civil Procedure 1908",
        "IEA": "Indian Evidence Act 1872 (mapped to BSA 2023)",
        "HMA": "Hindu Marriage Act 1955",
        "IDA": "Indian Divorce Act 1869",
        "MVA": "Motor Vehicles Act 1988",
        "NIA": "Negotiable Instruments Act 1881",
    }

    _JSON_ACT_MAP = {
        "ipc": "Indian Penal Code 1860 (mapped to BNS 2023)",
        "crpc": "Code of Criminal Procedure 1973 (mapped to BNSS 2023)",
        "cpc": "Code of Civil Procedure 1908",
        "coi": "Constitution of India 1950",
        "iea": "Indian Evidence Act 1872 (mapped to BSA 2023)",
        "hma": "Hindu Marriage Act 1955",
        "ida": "Indian Divorce Act 1869",
        "mva": "Motor Vehicles Act 1988",
        "nia": "Negotiable Instruments Act 1881",
    }

    _REGISTRY_PATH = resolve("data/laws") / "act_registry.json"
    _REGISTRY_VERSION = "v1"

    def _load_act_registry(self) -> Dict[str, dict]:
        """Load data/laws/act_registry.json into {canonical, alias} indexes.

        The alias index maps every act_name string the pipeline produces
        (including '(mapped to BNS 2023)' and '(Section Notes)' variants) to its
        registry entry so status metadata can be attached to every Document.
        """
        if getattr(self, "_registry_index", None) is not None:
            return self._registry_index
        self._registry_index = {"canonical": {}, "alias": {}}
        if not self._REGISTRY_PATH.exists():
            print(f"[WARN] Act registry not found at {self._REGISTRY_PATH}; acts will default to 'active'")
            return self._registry_index
        try:
            data = json.loads(self._REGISTRY_PATH.read_text(encoding="utf-8"))
            for canonical, entry in (data.get("acts") or {}).items():
                self._registry_index["canonical"][canonical] = entry
                for alias in [canonical] + list(entry.get("aliases", [])):
                    self._registry_index["alias"][alias] = entry
        except Exception as e:
            print(f"[WARN] Failed to load act registry {self._REGISTRY_PATH}: {e}")
        return self._registry_index

    def _registry_entry(self, act_name: str) -> dict:
        """Look up an act's registry entry by its pipeline act_name string."""
        if not act_name:
            return {}
        idx = self._load_act_registry()
        entry = idx["alias"].get(act_name)
        if entry:
            return entry
        # Fallback: strip parenthetical suffixes ('(mapped to BNS 2023)',
        # '(Guide)', '(Section Notes)') and match the canonical base name.
        base = re.sub(r"\s*\(.*\)\s*$", "", act_name).strip().lower()
        if base:
            for alias, ent in idx["alias"].items():
                if alias.lower() == base or alias.lower().endswith(" " + base) or alias.lower().startswith(base + " "):
                    return ent
        return {}

    def _annotate_status(self, metadata: dict) -> dict:
        """Attach legal-status metadata (status/effective dates/replaced_by/
        jurisdiction/source_type) to a Document's metadata dict."""
        act_name = metadata.get("act_name") or metadata.get("source_act") or ""
        entry = self._registry_entry(act_name)
        if not entry:
            metadata.setdefault("status", "active")
            metadata.setdefault("source_type", "primary")
            metadata.setdefault("jurisdiction", "Pan-India")
            return metadata
        metadata["status"] = entry.get("status", "active")
        metadata["source_type"] = entry.get("source_type", "primary")
        metadata["jurisdiction"] = entry.get("jurisdiction", "Pan-India")
        if entry.get("effective_from"):
            metadata["effective_from"] = entry["effective_from"]
        if entry.get("effective_until"):
            metadata["effective_until"] = entry["effective_until"]
        if entry.get("replaced_by"):
            metadata["replaced_by"] = entry["replaced_by"]
        if entry.get("replaces"):
            metadata["replaces"] = ", ".join(entry["replaces"])
        return metadata

    @staticmethod
    def _clean_control_chars(text: str) -> str:
        """Strip C0 control characters (keep \t\n\r) and U+FFFD replacement
        chars so corrupt extraction bytes never reach the LLM."""
        if not text:
            return text
        cleaned = "".join(ch for ch in text if ch >= " " or ch in "\t\n\r")
        return cleaned.replace("\ufffd", "")

    @staticmethod
    def _embedding_header(metadata: dict, text: str) -> str:
        """Prepend a compact act/section/topic/concept header so the vector
        embeddings see legal context rather than raw statute text alone."""
        lines = []
        for label, key in (("ACT", "act_name"), ("DOMAIN", "topic"),
                           ("SECTION", "section_number"), ("TITLE", "section_title"),
                           ("LEGAL CONCEPTS", "keywords")):
            val = metadata.get(key)
            if val:
                lines.append(f"{label}: {val}")
        if not lines:
            return text
        return "\n".join(lines) + "\n\n" + text

    def __init__(self,
                 pdf_directory: str = None,
                 chroma_db_path: str = None,
                 ollama_base_url: str = None):
        from dotenv import load_dotenv
        load_dotenv(override=True)

        self.pdf_directory = str(resolve(pdf_directory or os.getenv("PDF_DIRECTORY") or RAG_CONFIG["pdf_directory"]))
        self.chroma_db_path = str(resolve(chroma_db_path or os.getenv("CHROMA_DB_PATH") or RAG_CONFIG["chroma_db_path"]))
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        os.environ["CHROMA_TELEMETRY"] = "false"
        os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"

        self.embeddings = _OllamaEmbedder(self.ollama_base_url)

        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self._reranker = None
        self._all_docs_cache = None
        self._content_to_index = {}
        self._score_pool_cache = OrderedDict()
        self._last_ensemble_sig = None
        self._retrieve_lock = threading.Lock()

        self.reranker_enabled = RAG_CONFIG["reranker_enabled"]
        self.mmr_enabled = RAG_CONFIG["mmr_enabled"]
        self.dynamic_weights_enabled = RAG_CONFIG["dynamic_weights_enabled"]

        self._load_existing()

    def _load_existing(self):
        """Load persistent ChromaDB + BM25 retrievers from disk. Idempotent;
        safe to call again to (re)load retrievers on demand."""
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
                chunks_cache = resolve("data/chunks/bm25_docs.pkl")
                if chunks_cache.exists():
                    self._all_docs_cache = joblib.load(chunks_cache)
                    self._content_to_index = {
                        doc.page_content: i
                        for i, doc in enumerate(self._all_docs_cache)
                    }
                    try:
                        stored_count = self.vectorstore._collection.count()
                        if stored_count != len(self._all_docs_cache):
                            print(
                                f"[WARN] Vectorstore ({stored_count} docs) is out of "
                                f"sync with data/chunks/bm25_docs.pkl ({len(self._all_docs_cache)} docs). "
                                "Run build_knowledge_base() to rebuild."
                            )
                    except Exception:
                        pass
                    self.bm25_retriever = BM25Retriever.from_documents(
                        self._all_docs_cache,
                        preprocess_func=_bm25_tokenizer,
                        k=RAG_CONFIG["ensemble_top_k"]
                    )
                    self._build_ensemble("")
                    print("[OK] Loaded existing ChromaDB and BM25 retrievers from cache")
            except Exception as e:
                print(f"[WARN] Failed to load existing retrievers: {e}")

    def _corpus_signature(self) -> str:
        """Fingerprint the ingestion sources + chunking config so build_knowledge_base
        can skip a full re-embed when nothing changed."""
        import hashlib
        hasher = hashlib.sha256()
        hasher.update(f"{self._REGISTRY_VERSION}:{RAG_CONFIG['chunk_size']}:{RAG_CONFIG['chunk_overlap']}:{self.embeddings.model}".encode())
        for base in (self.pdf_directory, str(resolve("data/laws"))):
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                path = os.path.join(base, name)
                if os.path.isfile(path):
                    st = os.stat(path)
                    hasher.update(f"{name}:{st.st_size}:{int(st.st_mtime)}".encode())
        db = str(resolve("data/laws") / "IndiaLaw.db")
        if os.path.isfile(db):
            st = os.stat(db)
            hasher.update(f"IndiaLaw.db:{st.st_size}:{int(st.st_mtime)}".encode())
        return hasher.hexdigest()

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
        """(Re)build ensemble retriever with current settings.

        Skips reconstruction when the (weights, search type, top_k) signature is
        unchanged so repeated queries don't rebuild the retriever every time.
        """
        if not self.vectorstore or not self.bm25_retriever:
            return

        chroma_kwargs = self._get_chroma_kwargs(top_k)
        weights = self._get_weights(query)
        search_type = "mmr" if self.mmr_enabled else "similarity"

        sig = (tuple(weights), search_type, top_k)
        if self.ensemble_retriever is not None and sig == self._last_ensemble_sig:
            return
        self._last_ensemble_sig = sig

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
            chunk.page_content = self._embedding_header(chunk.metadata, self._clean_control_chars(chunk.page_content))
            chunk.metadata = self._annotate_status(chunk.metadata)
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
            "BNSS_2023.pdf": "Bharatiya Nagarik Suraksha Sanhita 2023",
            "BSA_2023.pdf": "Bharatiya Sakshya Adhiniyam 2023",
            "IPC_1860.pdf": "Indian Penal Code 1860",
            "Consumer_Protection_Act_2019.pdf": "Consumer Protection Act 2019",
            "RTI_Act_2005.pdf": "Right to Information Act 2005",
            "IT_Act_2000.pdf": "Information Technology Act 2000",
            "payment_of_wages_act_1936.pdf": "Payment of Wages Act 1936",
            "Minimum_Wages_Act_1948.pdf": "Minimum Wages Act 1948",
            "Industrial_Disputes_Act_1947.pdf": "Industrial Disputes Act 1947",
            "Code_on_Wages_2019.pdf": "Code on Wages 2019",
            "Industrial_Relations_Code_2020.pdf": "Industrial Relations Code 2020",
            "Code_on_Social_Security_2020.pdf": "Code on Social Security 2020",
            "OSHWC_Code_2020.pdf": "Occupational Safety, Health and Working Conditions Code 2020",
            "Digital_Personal_Data_Protection_Act_2023.pdf": "Digital Personal Data Protection Act 2023",
            "Indian_Contract_Act_1872.pdf": "Indian Contract Act 1872",
            "Gujarat_Rent_Control_Act_1999.pdf": "Bombay Rents, Hotel and Lodging House Rates Control Act 1947 (Gujarat)",
            "Bombay_Rent_Act_1947_Gujarat.pdf": "Bombay Rents, Hotel and Lodging House Rates Control Act 1947 (Gujarat)",
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
        # Config-driven legal concepts (config/domain_config.json -> legal_concepts)
        try:
            from src.domain_config import get_config
            for concept, keywords in (get_config().get("legal_concepts") or {}).items():
                if any(kw in text_lower for kw in keywords):
                    concepts.append(concept)
        except Exception:
            pass
        return list(dict.fromkeys(concepts))

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

    def _load_json_laws(self, skip_stems=frozenset()) -> List[Document]:
        """Load structured JSON law files from data/laws/ into Documents.

        Stems already covered by IndiaLaw.db (in `skip_stems`, uppercase) are
        skipped so the same act's sections are not loaded twice from two
        sources.
        """
        laws_dir = resolve("data/laws")
        if not laws_dir.exists():
            print("  [JSON] No data/laws/ directory found")
            return []

        documents = []
        act_name_map = self._JSON_ACT_MAP

        for json_path in sorted(laws_dir.glob("*.json")):
            stem = json_path.stem.lower()
            if stem == "act_registry":
                continue
            if stem.upper() in skip_stems:
                print(f"  [JSON] Skipping {json_path.name} (already loaded from IndiaLaw.db)")
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

    def _db_covered_stems(self) -> set:
        """Return the set of uppercase act stems already loaded from IndiaLaw.db
        so the JSON loader can skip duplicates."""
        db_path = resolve("data/laws") / "IndiaLaw.db"
        if not db_path.exists():
            return set()
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            finally:
                conn.close()
            return {t for t in tables if t in self._DB_ACT_MAP}
        except Exception:
            return set()

    def _load_db_laws(self) -> List[Document]:
        """Load structured law sections from IndiaLaw.db (clean canonical source)"""
        db_path = resolve("data/laws") / "IndiaLaw.db"
        if not db_path.exists():
            print(f"  [DB] No IndiaLaw.db found at {db_path}")
            return []

        act_name_map = self._DB_ACT_MAP

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

                    metadata = self._annotate_status({
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
                    })

                    documents.append(Document(
                        page_content=self._embedding_header(metadata, self._clean_control_chars(section_desc)),
                        metadata=metadata
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
            doc = Document(
                page_content=self._clean_control_chars(text),
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
            doc.metadata = self._annotate_status(doc.metadata)
            doc.page_content = self._embedding_header(doc.metadata, doc.page_content)
            return doc

        # Constitution format: {"ArtNo", "Name", "ArtDesc"} or
        # {"ArtNo", "Name", "Clauses": [{"ClauseNo", "ClauseDesc", "SubClauses": [...]}]}
        if "ArtNo" in item:
            section_desc = (item.get("ArtDesc") or "").strip()
            if not section_desc:
                clause_parts = []
                for clause in item.get("Clauses") or []:
                    if not isinstance(clause, dict):
                        continue
                    desc = (clause.get("ClauseDesc") or "").strip()
                    if desc:
                        clause_parts.append(desc)
                    for sub in clause.get("SubClauses") or []:
                        if not isinstance(sub, dict):
                            continue
                        sub_no = (sub.get("SubClauseNo") or "").strip()
                        sub_desc = (sub.get("SubClauseDesc") or "").strip()
                        if sub_desc:
                            clause_parts.append(f"({sub_no}) {sub_desc}".strip())
                section_desc = " ".join(p for p in clause_parts if p).replace("\ufffd", "").strip()
            if not section_desc:
                return None
            doc = Document(
                page_content=self._clean_control_chars(section_desc),
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
            doc.metadata = self._annotate_status(doc.metadata)
            doc.page_content = self._embedding_header(doc.metadata, doc.page_content)
            return doc

        # Full format: {"chapter", "section", "section_title", "section_desc"}
        section_desc = item.get("section_desc") or item.get("description", "")
        if not section_desc:
            return None

        section_num = item.get("Section") or item.get("section", "")
        section_title = item.get("section_title") or item.get("title", "")
        chapter = item.get("chapter", "")

        section_label = f"Section {section_num}" if section_num else ""
        topic = self._detect_topic(section_desc + " " + section_title, act_name)

        doc = Document(
            page_content=self._clean_control_chars(section_desc),
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
        doc.metadata = self._annotate_status(doc.metadata)
        doc.page_content = self._embedding_header(doc.metadata, doc.page_content)
        return doc

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

        sig_file = resolve("data/chunks/corpus_signature.json")
        cached_pkl = resolve("data/chunks/bm25_docs.pkl")
        sig = self._corpus_signature()
        if cached_pkl.exists() and sig_file.exists():
            try:
                saved = json.loads(sig_file.read_text(encoding="utf-8"))
            except Exception:
                saved = None
            if saved == sig:
                self._load_existing()
                if self._all_docs_cache is not None:
                    print(f"[SKIP] Corpus unchanged ({len(self._all_docs_cache)} docs) — loaded from cache")
                    return self._all_docs_cache

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

        # Load structured laws from IndiaLaw.db + JSON files (skipping acts the
        # DB already provides so sections are not embedded twice).
        print("🔄 Loading structured laws from IndiaLaw.db...")
        db_docs = self._load_db_laws()
        print("🔄 Loading structured JSON laws...")
        json_docs = self._load_json_laws(skip_stems=self._db_covered_stems())
        all_docs = all_chunks + db_docs + json_docs

        if not all_docs:
            print("[WARN] No documents created. Add PDFs to data/pdfs or JSON files to data/laws.")
            return []

        chunks_cache_dir = resolve("data/chunks")
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

        # Refresh derived indexes/caches so post-build retrieval uses the new corpus.
        self._all_docs_cache = all_docs
        self._content_to_index = {doc.page_content: i for i, doc in enumerate(all_docs)}
        self._score_pool_cache = OrderedDict()
        self._last_ensemble_sig = None
        self._build_ensemble("", top_k=RAG_CONFIG["top_k_retrieval"])
        print("✅ Ensemble retriever created")

        sig_file = resolve("data/chunks/corpus_signature.json")
        sig_file.write_text(json.dumps(self._corpus_signature()), encoding="utf-8")
        print("✅ Corpus signature saved (next build will skip re-embedding)")

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
        "hacked": "unauthorised access computer resource", "hacking": "unauthorised access computer resource", "hack": "unauthorised access computer resource",
        "website": "website computer resource", "website hacked": "unauthorised access computer resource",
        "online": "online computer resource", "account": "account computer resource",
        "identity theft": "identity theft", "impersonate": "cheating by personation", "impersonating": "cheating by personation",
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
        """Ensemble-equivalent combined relevance scores, keyed by corpus
        document index: 0.3 * vector relevance + 0.7 * BM25 (normalized).
        The EnsembleRetriever itself discards these scores (and its documents
        are object copies), so we recompute them to break title-boost ties
        deterministically. Results are LRU-cached per normalized query.
        """
        cached = self._score_pool_cache.get(query)
        if cached is not None:
            self._score_pool_cache.move_to_end(query)
            return cached

        scores = {}
        bm25 = getattr(getattr(self, "bm25_retriever", None), "vectorizer", None)
        if bm25 is not None and self._all_docs_cache:
            try:
                arr = bm25.get_scores(_bm25_tokenizer(query))
                mx = max(arr) if len(arr) else 0.0
                if mx > 0:
                    for i, val in enumerate(arr):
                        scores[i] = 0.7 * val / mx
            except Exception:
                pass
        try:
            hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=RAG_CONFIG["ensemble_top_k"])
            for d, s in hits:
                idx = self._content_to_index.get(d.page_content)
                if idx is not None:
                    scores[idx] = scores.get(idx, 0.0) + 0.3 * max(0.0, float(s))
        except Exception:
            pass

        self._score_pool_cache[query] = scores
        if len(self._score_pool_cache) > 32:
            self._score_pool_cache.popitem(last=False)
        return scores

    def _normalize_query(self, query: str) -> str:
        """Map common user phrasing to legal terms for better BM25 retrieval.

        Phrases are applied longest-first so specific mappings (e.g.
        "not paying my salary") aren't shadowed by shorter generic ones
        (e.g. "not paying" / "salary") that happen to be listed first.
        """
        q = query.lower()
        for phrase, replacement in sorted(
            self._QUERY_SYNONYMS.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            q = re.sub(r"\b" + re.escape(phrase) + r"\b", replacement, q)
        return q

    def retrieve_with_metadata(self, query: str, top_k: int = None) -> List[Dict]:
        if not self.ensemble_retriever:
            raise Exception("Knowledge base not built. Call build_knowledge_base() first.")

        top_k = top_k or RAG_CONFIG["top_k_retrieval"]
        ensemble_top_k = RAG_CONFIG["ensemble_top_k"]

        # Normalize user phrasing to legal terms (helps BM25 match statute text)
        retrieval_query = self._normalize_query(query)

        # HyDE is disabled (local LLM removed); dense query = normalized query.
        dense_query = retrieval_query

        # Rebuild ensemble with dynamically chosen weights + MMR settings.
        # Locked so concurrent requests can't rebuild the shared retriever
        # mid-retrieval (the signature check in _build_ensemble makes rebuilds rare).
        with self._retrieve_lock:
            self._build_ensemble(retrieval_query, top_k=ensemble_top_k)
            try:
                # Retrieve
                docs = self.ensemble_retriever.get_relevant_documents(dense_query)
            except Exception as e:
                # Embeddings unavailable (e.g. Ollama down) -> BM25-only fallback.
                print(f"  [Retrieval] Dense ensemble failed ({e}); falling back to BM25")
                docs = self.bm25_retriever.get_relevant_documents(dense_query)

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
                        pool_scores.get(self._content_to_index.get(d.page_content), 0.0),
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
            results.append(self._doc_to_source(doc, i))

        return results

    def _doc_to_source(self, doc: Document, index: int) -> Dict:
        """Convert a retrieved Document into the source-dict shape consumed by
        the domain filter, LLM context, and response formatter."""
        return {
            "index": index,
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
            "metadata": doc.metadata,
            "status": doc.metadata.get("status", "active"),
            "effective_from": doc.metadata.get("effective_from", ""),
            "effective_until": doc.metadata.get("effective_until", ""),
            "replaced_by": doc.metadata.get("replaced_by", ""),
        }

    def retrieve_by_act(self, query: str, act_name: str, top_k: int = 3) -> List[Dict]:
        """Return top_k sources from a single act, ranked by lexical overlap
        with the query. Used to pull a repealed act's replacement law into the
        context without a second embedding pass."""
        if not self._all_docs_cache:
            raise Exception("Knowledge base not built. Call build_knowledge_base() first.")

        entry = self._load_act_registry()["canonical"].get(act_name)
        aliases = {a.lower() for a in ([act_name] + list(entry.get("aliases", [])))} if entry else {act_name.lower()}
        pool = [d for d in self._all_docs_cache
                if (d.metadata.get("act_name") or "").lower() in aliases]
        if not pool:
            return []

        terms = {w for w in re.findall(r"[a-z0-9]{4,}", self._normalize_query(query).lower())}

        def _score(d: Document) -> tuple:
            title = (d.metadata.get("section_title") or "").lower()
            text = (d.page_content + " " + title).lower()
            title_hits = sum(1 for t in terms if t in title)
            body_hits = sum(1 for t in terms if t in text)
            return (title_hits * 3 + body_hits, len(d.page_content))

        pool.sort(key=_score, reverse=True)

        seen = set()
        out = []
        for d in pool:
            key = (d.metadata.get("act_name"), d.metadata.get("section_number"))
            if key in seen:
                continue
            seen.add(key)
            out.append(self._doc_to_source(d, len(out)))
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def _split_field(val) -> list:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return []

    def retrieve_dense(self, query: str, domain: str, top_k: int = 4) -> List[Dict]:
        if not self.ensemble_retriever:
            self._load_existing()
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
