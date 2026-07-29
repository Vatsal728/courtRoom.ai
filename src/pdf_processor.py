import os
import fitz
from pathlib import Path
import json
import re
from typing import List, Dict

class PDFProcessor:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error extracting {pdf_path}: {e}")
        return text
    
    def chunk_text(self, text: str, source: str, domain: str) -> List[Dict]:
        """Split text into overlapping chunks with RICH metadata"""
        words = text.split()
        chunks = []
        
        current_section = "Preamble"
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 50:
                continue
            
            chunk_text = " ".join(chunk_words)
            
            # Extract section number if present
            section_match = re.search(r'[Ss]ection\s+(\d+)', chunk_text)
            if section_match:
                current_section = f"Section {section_match.group(1)}"
            
            chunks.append({
                "text": chunk_text,
                "source": source,
                "domain": domain,
                "chunk_id": len(chunks),
                "word_count": len(chunk_words),
                "section": current_section,  # ← NEW
                "topic": self._extract_topic(chunk_text),  # ← NEW
                "applicable_courts": self._extract_courts(domain),  # ← NEW
                "relevance_keywords": self._extract_keywords(chunk_text)  # ← NEW
            })
        
        return chunks

    def _extract_topic(self, text: str) -> str:
        """Extract main topic from chunk"""
        first_sentence = text.split('.')[0]
        return first_sentence[:100] if first_sentence else "General"

    def _extract_courts(self, domain: str) -> List[str]:
        """Get applicable courts for domain"""
        courts_map = {
            "consumer": ["District Consumer Commission", "State Consumer Commission"],
            "labour": ["Labour Commissioner", "Industrial Tribunal"],
            "criminal": ["District Court", "High Court"],
            "rti": ["Central Information Commission"],
            "rent": ["District Court", "Rent Control Board"],
            "cyber": ["Cybercrime Police", "District Court"]
        }
        return courts_map.get(domain, ["District Court"])

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(keywords))[:5]  # Top 5 unique keywords
    
    def process_all_pdfs(self, pdf_dir: str = "data/pdfs") -> Dict:
        """Process all PDFs and return chunks grouped by domain"""
        domain_mapping = {
            "IPC_1860.pdf": "criminal",
            "BNS_2023.pdf": "criminal",
            "Consumer_Protection_Act_2019.pdf": "consumer",
            "RTI_Act_2005.pdf": "rti",
            "Payment_of_Wages_Act_1936.pdf": "labour",
            "Minimum_Wages_Act_1948.pdf": "labour",
            "IT_Act_2000.pdf": "cyber",
            "Gujarat_Rent_Control_Act_1999.pdf": "rent"
        }
        
        all_chunks = {"consumer": [], "labour": [], "criminal": [], "rti": [], "rent": [], "cyber": []}
        
        for pdf_file in Path(pdf_dir).glob("*.pdf"):
            domain = domain_mapping.get(pdf_file.name)
            if not domain:
                continue
            print(f"Processing {pdf_file.name}...")
            
            text = self.extract_text(str(pdf_file))
            chunks = self.chunk_text(text, pdf_file.name, domain)
            all_chunks[domain].extend(chunks)
            
            print(f"  [OK] {len(chunks)} chunks created for {domain}")
        
        # Summary
        total = sum(len(chunks) for chunks in all_chunks.values())
        print(f"\nSummary - Total chunks: {total}")
        for domain, chunks in all_chunks.items():
            if chunks:
                print(f"  - {domain}: {len(chunks)} chunks")
        
        return all_chunks

if __name__ == "__main__":
    processor = PDFProcessor(chunk_size=512, overlap=50)
    chunks = processor.process_all_pdfs()
    
    # Save chunks for Phase 6 (RAG)
    os.makedirs("data/chunks", exist_ok=True)
    with open("data/chunks/all_chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("\n[OK] Chunks saved to data/chunks/all_chunks.json")
