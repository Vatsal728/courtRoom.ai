import re
from typing import Dict, List, Tuple

try:
    import spacy
except ImportError:
    spacy = None

try:
    from nltk import pos_tag, ne_chunk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk
    _HAS_NLTK = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    _HAS_NLTK = False


class NLPPipeline:
    def __init__(self):
        self._nlp = None
        self.stop_words = set(stopwords.words('english')) if _HAS_NLTK else set()
        self.domain_keywords = {
            "consumer": ["defective", "refund", "product", "order", "delivery", "quality", "complaint", "warranty", "online"],
            "labour": ["salary", "wages", "employer", "employee", "fired", "terminated", "resignation", "paycheck", "promotion"],
            "criminal": ["theft", "assault", "harassment", "cheating", "fraud", "violence", "crime", "police", "fir"],
            "rti": ["information", "government", "public", "document", "records", "pio", "request", "disclosure"],
            "rent": ["landlord", "tenant", "eviction", "deposit", "rent", "lease", "property", "maintenance"],
            "cyber": ["hacking", "password", "account", "fraud", "phishing", "data", "online", "website", "email"]
        }

    @property
    def nlp(self):
        if self._nlp is None and spacy is not None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                try:
                    spacy.cli.download("en_core_web_sm")
                    self._nlp = spacy.load("en_core_web_sm")
                except Exception:
                    pass
        return self._nlp

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        if self.nlp is not None:
            doc = self.nlp(text)
            tokens = [t.text for t in doc if not t.is_stop and len(t.text) > 2]
        elif _HAS_NLTK:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        else:
            tokens = re.findall(r'\b[a-z]{3,}\b', text)
        return tokens

    def pos_tagging(self, tokens: List[str]) -> List[Tuple[str, str]]:
        if _HAS_NLTK:
            return pos_tag(tokens)
        return [(t, "UNKNOWN") for t in tokens]

    def ner(self, text: str) -> List[Dict]:
        if self.nlp is not None:
            doc = self.nlp(text)
            return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        tokens = self.tokenize(text)
        if _HAS_NLTK:
            try:
                tree = ne_chunk(pos_tag(tokens))
                entities = []
                for subtree in tree:
                    if hasattr(subtree, 'label'):
                        entities.append({
                            "text": " ".join(w for w, _ in subtree),
                            "label": subtree.label()
                        })
                return entities
            except Exception:
                pass
        return []

    def extract_entities(self, text: str) -> Dict:
        tokens = self.tokenize(text)
        entities = {
            "amounts": [],
            "people": [],
            "organizations": [],
            "durations": [],
            "named_entities": []
        }

        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "MONEY" or ent.label_ == "CARDINAL":
                    entities["amounts"].append(ent.text)
                elif ent.label_ == "PERSON":
                    entities["people"].append(ent.text)
                elif ent.label_ == "ORG":
                    entities["organizations"].append(ent.text)
                elif ent.label_ in ("DATE", "TIME"):
                    entities["durations"].append(ent.text)
                entities["named_entities"].append({"text": ent.text, "label": ent.label_})
        elif _HAS_NLTK:
            pos_tags = pos_tag(tokens)
            for token, tag in pos_tags:
                if tag == "CD":
                    entities["amounts"].append(token)
                elif tag == "NNP":
                    entities["people"].append(token)
                elif tag == "NNPS":
                    entities["organizations"].append(token)

        time_words = ["day", "week", "month", "year", "hour", "minute"]
        for t in tokens:
            if t in time_words:
                entities["durations"].append(t)

        return entities

    def detect_intent(self, text: str) -> Dict:
        tokens = self.tokenize(text)
        scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for token in tokens if token in keywords)
            if score > 0:
                scores[domain] = score

        if scores:
            top_domain = max(scores, key=scores.get)
            confidence = scores[top_domain] / max(len(tokens), 1)
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
        tokens = self.tokenize(text)
        entities = self.extract_entities(text)
        intent = self.detect_intent(text)

        return {
            "original_text": text,
            "tokens": tokens,
            "token_count": len(tokens),
            "entities": entities,
            "intent": intent
        }


if __name__ == "__main__":
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
