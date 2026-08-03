"""
config.py - Central configuration for courtRoom.ai
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── RAG Pipeline ────────────────────────────────────────────────

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

    # Dynamic weights (auto-detect section-number queries → favor BM25)
    "dynamic_weights_enabled": True,

    # MMR (Maximum Marginal Relevance)
    "mmr_enabled": False,
    "mmr_fetch_k": 20,
    "mmr_lambda_mult": 0.7,

    # Reranker (BGE Cross-Encoder)
    "reranker_enabled": False,
    "reranker_model": "D:/courtRoom.ai/.hf_cache/local/bge-reranker-v2-m3",
    "reranker_device": "cpu",

    # HyDE (Hypothetical Document Embeddings)
    "hyde_enabled": False,
    "hyde_max_tokens": 200,
    "hyde_timeout": 30,
}

# ── LLM ─────────────────────────────────────────────────────────

LLM_CONFIG = {
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    "ollama_embed_model": "nomic-embed-text",
    "request_timeout": int(os.getenv("OLLAMA_TIMEOUT", "300")),
    "temperature": 0.1,
    "max_tokens": 2048,
}

# ── MongoDB ─────────────────────────────────────────────────────

MONGODB_CONFIG = {
    "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
    "db_name": os.getenv("MONGODB_DB", "courtroom_ai"),
    "collections": {
        "queries": "queries",
        "pdfs": "pdfs",
        "users": "users",
        "cases": "cases",
        "consultations": "consultations",
    }
}

# ── Classifier ──────────────────────────────────────────────────

CLASSIFIER_CONFIG = {
    "min_confidence": 0.3,
    "high_confidence_threshold": 0.8,
    "enable_secondary_detection": True,
}

# ─── Response Formatter ─────────────────────────────────────────

FORMATTER_CONFIG = {
    "include_short_answer": True,
    "include_criminal_route": True,
    "include_civil_route": True,
    "include_practical_steps": True,
    "include_compensation": True,
    "max_sources": 5,
    "max_remedies": 4,
}

# ── API ─────────────────────────────────────────────────────────

API_CONFIG = {
    "host": os.getenv("API_HOST", "127.0.0.1"),
    "port": int(os.getenv("API_PORT", "8000")),
    "cors_origins": os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173").split(","),
    "rate_limit_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")),
    "max_query_length": int(os.getenv("MAX_QUERY_LENGTH", "2000")),
}

# ── Caching ─────────────────────────────────────────────────────

CACHE_CONFIG = {
    "enabled": True,
    "maxsize": 100,
    "ttl": 3600,  # 1 hour
}

# ── ChromaDB ────────────────────────────────────────────────────

CHROMA_CONFIG = {
    "telemetry": False,
    "anonymized_telemetry": False,
    "persistent": True,
    "path": "chroma_db",
}
