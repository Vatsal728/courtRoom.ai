#!/usr/bin/env python
"""Build knowledge base from PDFs"""

import sys
sys.path.insert(0, 'D:\\courtRoom.ai')

from src.full_rag import FullRAGSystem

if __name__ == "__main__":
    rag = FullRAGSystem()
    rag.build_knowledge_base()
    print("\n✅ Knowledge base built successfully!")
