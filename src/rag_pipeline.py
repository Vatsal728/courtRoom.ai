"""
rag_pipeline.py - Enhanced RAG with better chunking and metadata
"""

import os
import sys
import io
import warnings
warnings.filterwarnings("ignore")

os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Set console output encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from typing import List, Dict, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.bm25 import BM25Retriever
import chromadb
from chromadb.config import Settings
import json
from pathlib import Path

class RAGPipeline:
    """Enhanced RAG with rich metadata and better retrieval"""
    
    def __init__(self, 
                 pdf_directory: str = None,
                 chroma_db_path: str = None,
                 ollama_base_url: str = None):
        """Initialize improved RAG pipeline"""
        from dotenv import load_dotenv
        load_dotenv()
        
        self.pdf_directory = pdf_directory or os.getenv("PDF_DIRECTORY", "data/pdfs")
        self.chroma_db_path = chroma_db_path or os.getenv("CHROMA_DB_PATH", "chroma_db")
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Disable ChromaDB telemetry
        os.environ["CHROMA_TELEMETRY"] = "false"
        os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
        
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=self.ollama_base_url
        )
        
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None

        # Auto-load existing database on startup if present
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
                # If we have serialized chunks, we can reconstruct the BM25 index
                chunks_cache = Path("data/chunks/all_chunks_langchain.json")
                if chunks_cache.exists():
                    with open(chunks_cache, "r", encoding="utf-8") as f:
                        serialized_data = json.load(f)
                    
                    # Reconstruct LangChain document objects
                    from langchain_core.documents import Document
                    all_docs = [
                        Document(page_content=item["page_content"], metadata=item["metadata"])
                        for item in serialized_data
                    ]
                    self.bm25_retriever = BM25Retriever.from_documents(all_docs)
                    
                    self.ensemble_retriever = EnsembleRetriever(
                        retrievers=[
                            self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                            self.bm25_retriever
                        ],
                        weights=[0.6, 0.4]
                    )
                    print("[OK] Loaded existing ChromaDB and BM25 retrievers from cache")
            except Exception as e:
                print(f"[WARN] Failed to load existing retrievers: {e}")
    
    def chunk_pdf_with_metadata(self, pdf_path: str) -> List[Dict]:
        """
        Enhanced chunking with rich metadata
        
        Metadata includes:
        - State/jurisdiction
        - Section number
        - Act name
        - Chunk topic
        - Related sections
        - Applicable courts
        - Case scenarios
        """
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Extract act name from filename
        act_name = os.path.basename(pdf_path).replace(".pdf", "").replace("_", " ")
        
        # State detection
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
        
        # Smart chunking strategy
        # Different chunk sizes for different types of content
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )
        
        chunks = splitter.split_documents(documents)
        
        # Enhance each chunk with metadata
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # Extract section number if present
            section_match = self._extract_section_number(chunk.page_content)
            
            # Detect topic/subject
            topic = self._detect_topic(chunk.page_content, act_name)
            
            # Get related sections
            related_sections = self._find_related_sections(chunk.page_content)
            
            # Determine applicable courts
            courts = self._determine_applicable_courts(act_name, topic)
            
            # Extract key concepts
            concepts = self._extract_key_concepts(chunk.page_content)
            
            # Create comprehensive metadata
            metadata = {
                "source": pdf_path,
                "act_name": act_name,
                "state": detected_state,
                "section_number": section_match or f"Section {i}",
                "topic": topic,
                "related_sections": related_sections,
                "applicable_courts": courts,
                "keywords": concepts,
                "page_number": chunk.metadata.get("page", i),
                "chunk_index": i,
                "case_scenarios": self._generate_case_scenarios(topic),
                "penalty_or_relief": self._extract_penalty_or_relief(chunk.page_content)
            }
            
            chunk.metadata.update(metadata)
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _extract_section_number(self, text: str) -> Optional[str]:
        """Extract section numbers from text"""
        import re
        
        patterns = [
            r"Section\s+(\d+[A-Z]*)",
            r"Article\s+(\d+)",
            r"Rule\s+(\d+)",
            r"Schedule\s+(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _detect_topic(self, text: str, act_name: str) -> str:
        """Detect topic/subject of chunk"""
        
        topic_keywords = {
            "Criminal": ["intimidation", "harassment", "threat", "assault", "offense", "crime"],
            "Civil": ["damages", "injunction", "relief", "compensation", "lawsuit"],
            "Eviction": ["eviction", "possession", "tenancy", "landlord", "tenant"],
            "Payment": ["wages", "salary", "payment", "refund", "compensation"],
            "Property": ["property", "land", "building", "premises", "possession"],
            "Family": ["marriage", "divorce", "custody", "maintenance", "inheritance"],
            "Labor": ["employment", "worker", "employee", "wages", "conditions"],
            "Contract": ["agreement", "contract", "breach", "performance", "terms"],
            "Cyber": ["online", "digital", "internet", "email", "data", "cyber"],
            "Consumer": ["consumer", "goods", "service", "defective", "refund"]
        }
        
        text_lower = text.lower()
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        
        return "General Legal"
    
    def _find_related_sections(self, text: str) -> List[str]:
        """Find related sections mentioned in text"""
        import re
        
        pattern = r"(?:refer to |see |under |pursuant to |as per ).*?(?:Section|Article|Rule)\s+(\d+[A-Z]*)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(set(matches))[:5]
    
    def _determine_applicable_courts(self, act_name: str, topic: str) -> List[str]:
        """Determine which courts can handle this"""
        
        courts = set()
        
        # Default courts
        courts.add("District Court")
        
        # Topic-based courts
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
        """Extract key legal concepts"""
        
        concepts = []
        
        # Common legal concepts
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
        """Generate applicable case scenarios"""
        
        scenarios = {
            "Criminal": [
                "Intimidation and threats",
                "Harassment and abuse",
                "Wrongful restraint",
                "Breach of peace",
                "Assault"
            ],
            "Civil": [
                "Contract breach",
                "Damages recovery",
                "Injunction relief",
                "Specific performance",
                "Property disputes"
            ],
            "Eviction": [
                "Illegal eviction attempt",
                "Wrongful lock-out",
                "Utility disconnection",
                "Non-payment of rent",
                "Breach of tenancy"
            ],
            "Labor": [
                "Wrongful termination",
                "Non-payment of wages",
                "Unsafe working conditions",
                "Discrimination",
                "Sexual harassment"
            ]
        }
        
        return scenarios.get(topic, ["General legal matter"])
    
    def _extract_penalty_or_relief(self, text: str) -> Optional[str]:
        """Extract penalty or relief provisions"""
        import re
        
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
    
    def build_knowledge_base(self):
        """Build complete knowledge base from PDFs"""
        
        print("🔄 Building knowledge base...")
        
        all_chunks = []
        
        # Load and chunk all PDFs
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
        
        if not all_chunks:
            print("[WARN] No chunks created. Add PDF files to data/pdfs first.")
            return []
        
        # Serialize chunks cache for later reconstruction
        chunks_cache_dir = Path("data/chunks")
        chunks_cache_dir.mkdir(parents=True, exist_ok=True)
        serialized_chunks = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in all_chunks
        ]
        with open(chunks_cache_dir / "all_chunks_langchain.json", "w", encoding="utf-8") as f:
            json.dump(serialized_chunks, f, ensure_ascii=False, indent=2)
        
        # Create ChromaDB vector store
        print("🔄 Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.chroma_db_path,
            client_settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        self.vectorstore.persist()
        print("✅ Vector store created")
        
        # Create BM25 retriever
        print("🔄 Creating BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(all_chunks)
        print("✅ BM25 retriever created")
        
        # Create ensemble retriever
        print("🔄 Creating ensemble retriever...")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]
        )
        print("✅ Ensemble retriever created")
        
        return all_chunks
    
    def retrieve_with_metadata(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents with rich metadata"""
        
        if not self.ensemble_retriever:
            raise Exception("Knowledge base not built. Call build_knowledge_base() first.")
        
        # Retrieve documents
        docs = self.ensemble_retriever.get_relevant_documents(query)
        
        # Extract and organize metadata
        results = []
        for i, doc in enumerate(docs[:top_k]):
            result = {
                "index": i,
                "content": doc.page_content,
                "text": doc.page_content,  # Compatibility field
                "act_name": doc.metadata.get("act_name", "Unknown"),
                "section_number": doc.metadata.get("section_number", "Unknown"),
                "state": doc.metadata.get("state", "Pan-India"),
                "topic": doc.metadata.get("topic", "General"),
                "applicable_courts": doc.metadata.get("applicable_courts", ["District Court"]),
                "keywords": doc.metadata.get("keywords", []),
                "related_sections": doc.metadata.get("related_sections", []),
                "case_scenarios": doc.metadata.get("case_scenarios", []),
                "penalty_or_relief": doc.metadata.get("penalty_or_relief", None),
                "page_number": doc.metadata.get("page_number", 0),
                "source_act": doc.metadata.get("act_name", "Unknown"),
                "courts": doc.metadata.get("applicable_courts", ["District Court"]),
                "metadata": doc.metadata  # Compatibility field
            }
            results.append(result)
        
        return results
    
    def retrieve_dense(self, query: str, domain: str, top_k: int = 4) -> List[Dict]:
        """Compatibility method mapping retrieve_dense requests"""
        if not self.ensemble_retriever:
            # Fallback to reload database
            self.__init__(chroma_db_path=self.chroma_db_path, ollama_base_url=self.ollama_base_url)
        
        try:
            return self.retrieve_with_metadata(query, top_k=top_k)
        except Exception:
            return []
            
    def search_by_jurisdiction(self, query: str, state: str, top_k: int = 5) -> List[Dict]:
        """Search filtered by state/jurisdiction"""
        
        all_results = self.retrieve_with_metadata(query, top_k=20)
        
        # Filter by state
        filtered = [r for r in all_results if state.lower() in r["state"].lower() or r["state"] == "Pan-India"]
        
        return filtered[:top_k]
    
    def search_by_topic(self, query: str, topic: str, top_k: int = 5) -> List[Dict]:
        """Search filtered by topic"""
        
        all_results = self.retrieve_with_metadata(query, top_k=20)
        
        # Filter by topic
        filtered = [r for r in all_results if topic.lower() in r["topic"].lower()]
        
        return filtered[:top_k]
    
    def get_related_laws(self, section: str) -> List[Dict]:
        """Get related laws for a section"""
        
        query = f"Related to {section}"
        results = self.retrieve_with_metadata(query, top_k=10)
        
        # Filter for related sections
        related = [
            r for r in results 
            if section in r.get("related_sections", []) or section in r.get("keywords", [])
        ]
        
        return related[:5]


# Initialize on module load
if __name__ == "__main__":
    # Test
    rag = RAGPipeline()
    chunks = rag.build_knowledge_base()
    
    # Test retrieval
    query = "I rented a flat but landlord is cutting water supply"
    results = rag.retrieve_with_metadata(query)
    
    print("\n=== RETRIEVAL RESULTS ===")
    for r in results[:3]:
        print(f"\n📄 {r['act_name']} - {r['section_number']}")
        print(f"   Topic: {r['topic']}")
        print(f"   Courts: {', '.join(r['applicable_courts'])}")
        print(f"   State: {r['state']}")
