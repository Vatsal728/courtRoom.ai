import joblib
from pathlib import Path
from typing import Dict, List

class CaseTypeClassifier:
    def __init__(self):
        """Load pre-trained models"""
        models_dir = Path("models")
        self.clf = joblib.load(models_dir / "nb_classifier.pkl")
        self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
        self.domains = self.clf.classes_
    
    def predict(self, text: str) -> Dict:
        """Predict domain and confidence for a query"""
        # Vectorize
        X = self.vectorizer.transform([text])
        
        # Get probabilities
        probs = self.clf.predict_proba(X)[0]
        
        # Create probability dict
        domain_probs = {domain: float(prob) for domain, prob in zip(self.domains, probs)}
        
        # Get top domain
        top_domain = self.clf.predict(X)[0]
        top_confidence = domain_probs[top_domain]
        
        return {
            "primary_domain": top_domain,
            "confidence": top_confidence,
            "all_probabilities": domain_probs
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

if __name__ == "__main__":
    # Test
    classifier = CaseTypeClassifier()
    
    test_queries = [
        "My salary has not been paid for 3 months",
        "I bought a defective product online",
        "The landlord is evicting me illegally",
        "Someone hacked my PC and stole the valuable projects and bitcoins."
    ]
    
    for query in test_queries:
        result = classifier.predict(query)
        print(f"\nQuery: {query}")
        print(f"   Domain: {result['primary_domain']} ({result['confidence']:.2%})")
        print(f"   All probabilities:")
        for domain, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"     - {domain}: {prob:.2%}")
