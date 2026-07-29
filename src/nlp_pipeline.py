import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
import re
from typing import Dict, List, Tuple

class NLPPipeline:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.stop_words = set(stopwords.words('english'))
        self.domain_keywords = {
            "consumer": ["defective", "refund", "product", "order", "delivery", "quality", "complaint", "warranty", "online"],
            "labour": ["salary", "wages", "employer", "employee", "fired", "terminated", "resignation", "paycheck", "promotion"],
            "criminal": ["theft", "assault", "harassment", "cheating", "fraud", "violence", "crime", "police", "fir"],
            "rti": ["information", "government", "public", "document", "records", "pio", "request", "disclosure"],
            "rent": ["landlord", "tenant", "eviction", "deposit", "rent", "lease", "property", "maintenance"],
            "cyber": ["hacking", "password", "account", "fraud", "phishing", "data", "online", "website", "email"]
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Clean text
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        return tokens
    
    def pos_tagging(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """POS tag tokens"""
        return pos_tag(tokens)
    
    def ner(self, tokens: List[str]) -> nltk.Tree:
        """Named entity recognition"""
        pos_tags = pos_tag(tokens)
        return ne_chunk(pos_tags)
    
    def extract_entities(self, text: str) -> Dict:
        """Extract named entities and amounts"""
        tokens = self.tokenize(text)
        entities = {
            "amounts": [],
            "people": [],
            "organizations": [],
            "durations": []
        }
        
        pos_tags = pos_tag(tokens)
        
        # Extract amounts (CD = cardinal number)
        for token, tag in pos_tags:
            if tag == "CD":
                entities["amounts"].append(token)
        
        # Extract proper nouns (person/org)
        for token, tag in pos_tags:
            if tag == "NNP":
                entities["people"].append(token)
            elif tag == "NNPS":
                entities["organizations"].append(token)
        
        # Extract time expressions
        time_words = ["day", "week", "month", "year", "hour", "minute"]
        for token, tag in pos_tags:
            if token in time_words:
                entities["durations"].append(token)
        
        return entities
    
    def detect_intent(self, text: str) -> Dict:
        """Detect legal domain from text"""
        tokens = self.tokenize(text)
        scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for token in tokens if token in keywords)
            if score > 0:
                scores[domain] = score
        
        # Return domain with highest score
        if scores:
            top_domain = max(scores, key=scores.get)
            confidence = scores[top_domain] / len(tokens)
            return {
                "domain": top_domain,
                "confidence": min(confidence, 1.0),
                "all_scores": scores
            }
        
        return {
            "domain": "other",
            "confidence": 0.0,
            "all_scores": {}
        }
    
    def process_query(self, text: str) -> Dict:
        """Full NLP pipeline on user query"""
        tokens = self.tokenize(text)
        entities = self.extract_entities(text)
        intent = self.detect_intent(text)
        
        return {
            "original_text": text,
            "tokens": tokens,
            "token_count": len(tokens),
            "entities": entities,
            "intent": intent,
            "pos_tags": pos_tag(tokens)[:10]  # First 10 for display
        }

if __name__ == "__main__":
    # Test
    pipeline = NLPPipeline()
    
    test_queries = [
        "My employer did not pay my salary for 3 months",
        "I bought a defective product online and want a refund",
        "The landlord is illegally evicting me"
    ]
    
    for query in test_queries:
        result = pipeline.process_query(query)
        print(f"\nQuery: {query}")
        print(f"   Domain: {result['intent']['domain']} ({result['intent']['confidence']:.2%})")
        print(f"   Entities: {result['entities']}")
