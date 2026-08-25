"""
user_docs_service.py - Per-user document RAG for uploaded files.

PDF uploads are chunked, embedded with the configured provider (Google Gemini
by default), and stored in a per-user Chroma namespace so the chat can answer
from the user's own documents without touching the master knowledge base.

Storage:
  - Vectors: storage/chroma_db_user/<user_id>/ (raw chromadb, cosine space)
  - Metadata: MongoDB collection `user_docs`
"""
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pymongo import MongoClient

from src.embedding_provider import EmbeddingProvider, get_embedding_provider
from src.paths import resolve

DOC_HEADER = "DOCUMENT: {filename}\n\n"

CHUNK_SIZE = int(os.getenv("USER_DOC_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("USER_DOC_CHUNK_OVERLAP", "200"))


class UserDocsService:
    def __init__(self, provider: Optional[EmbeddingProvider] = None, mongodb_uri: str = None,
                 mongodb_db: str = None):
        self.provider = provider or get_embedding_provider()
        self.base_dir = Path(str(resolve("storage/chroma_db_user")))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._mongo = MongoClient(mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017"))[
            mongodb_db or os.getenv("MONGODB_DB", "courtroom_ai")
        ]
        self._clients = {}
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _client(self, user_id: str) -> chromadb.ClientAPI:
        safe = "".join(c if c.isalnum() else "_" for c in user_id)[:64]
        if safe not in self._clients:
            user_dir = self.base_dir / safe
            user_dir.mkdir(parents=True, exist_ok=True)
            self._clients[safe] = chromadb.PersistentClient(path=str(user_dir))
        return self._clients[safe]

    def _collection(self, user_id: str):
        return self._client(user_id).get_or_create_collection(
            name="user_docs", metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _extract_text(file_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            text = "\n\n".join(p.page_content for p in pages)
        finally:
            os.unlink(tmp_path)
        return (text or "").replace("\ufffd", "")

    def _chunk_text(self, text: str) -> List[str]:
        splits = [c.page_content for c in self._splitter.create_documents([text])]
        return [s.strip() for s in splits if s.strip()]

    def ingest_pdf(self, user_id: str, file_bytes: bytes, filename: str) -> Dict:
        full_text = self._extract_text(file_bytes)
        if not full_text.strip():
            raise ValueError("Could not extract text from this PDF (scanned/empty?)")

        splits = self._chunk_text(full_text)
        if not splits:
            raise ValueError("No usable text chunks extracted from this PDF")

        header = DOC_HEADER.format(filename=filename)
        texts = [header + s for s in splits]
        embeddings = self.provider.embed_documents(texts)

        doc_id = uuid.uuid4().hex[:16]
        ids = [f"{doc_id}_{i}" for i in range(len(texts))]
        metadatas = [
            {
                "filename": filename,
                "doc_id": doc_id,
                "user_id": user_id,
                "chunk_index": i,
                "source": filename,
            }
            for i in range(len(texts))
        ]
        self._collection(user_id).upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )
        self._mongo.user_docs.insert_one({
            "user_id": user_id,
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": len(texts),
            "provider": self.provider.name,
            "model": getattr(self.provider, "model", ""),
            "created_at": datetime.now(),
        })
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": len(texts),
            "provider": self.provider.name,
        }

    def search(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        collection = self._collection(user_id)
        count = collection.count()
        if count == 0:
            return []
        qvec = self.provider.embed_query(query)
        res = collection.query(
            query_embeddings=[qvec], n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        results = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, doc in enumerate(docs):
            meta = metas[i] or {}
            results.append({
                "rank": i + 1,
                "text": doc,
                "similarity": round(max(0.0, 1.0 - float(dists[i])), 4),
                "filename": meta.get("filename"),
                "doc_id": meta.get("doc_id"),
                "chunk_index": meta.get("chunk_index"),
            })
        return results

    def list_docs(self, user_id: str) -> List[Dict]:
        docs = list(self._mongo.user_docs.find({"user_id": user_id}).sort("created_at", -1))
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs
