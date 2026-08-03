import re
from typing import Tuple, List

NEGATION_PATTERNS = [
    r"\b(?:not|no|never|don't|doesn't|didn't|wasn't|won't)\s+\w*(?:criminal|crime|illegal|theft|assault|fraud|harassment|defame|hack)\b",
]

class DomainClassifier:
    """Classify query into legal domain with NaiveBayes + keyword ensemble"""

    CRIMINAL_KEYWORDS = [
        "fir", "police", "crime", "criminal", "arrest", "jail", "prison",
        "intimidation", "threat", "harassment", "assault", "hit", "beat",
        "defame", "slander", "libel", "blackmail", "extortion", "rape",
        "theft", "robbery", "murder", "poison", "knife", "gun", "weapon"
    ]
    CIVIL_KEYWORDS = [
        "injunction", "damages", "compensation", "suit", "claim", "relief",
        "civil court", "breach", "contract", "agreement", "money", "dues",
        "settlement", "judgment", "decree", "recovery", "property", "title"
    ]
    RENT_KEYWORDS = [
        "rent", "landlord", "tenant", "eviction", "lease", "deposit",
        "flat", "house", "apartment", "property", "notice", "possession"
    ]
    LABOR_KEYWORDS = [
        "employment", "employer", "employee", "wages", "salary", "leave",
        "dismissal", "termination", "redundancy", "layoff", "work", "job",
        "working", "hours", "shift", "gratuity", "bonus", "pf", "esi"
    ]
    FAMILY_KEYWORDS = [
        "marriage", "divorce", "custody", "maintenance", "alimony",
        "inheritance", "succession", "will", "dowry", "domestic", "abuse",
        "child", "parent", "adoption", "guardianship", "separation"
    ]
    DEFAMATION_KEYWORDS = [
        "defame", "slander", "libel", "false", "reputation", "character",
        "rumors", "gossip", "social media", "facebook", "instagram",
        "twitter", "post", "comment", "viral", "humiliation"
    ]
    CYBER_KEYWORDS = [
        "online", "internet", "website", "email", "password", "hacking",
        "cyber", "digital", "data", "privacy", "social media", "whatsapp",
        "screenshot", "viral", "deepfake", "malware", "ransomware"
    ]
    COMMERCIAL_KEYWORDS = [
        "business", "company", "partnership", "trader", "goods", "service",
        "customer", "client", "vendor", "supplier", "invoice", "payment",
        "franchise", "patent", "trademark", "copyright", "intellectual"
    ]

    DOMAIN_KEYWORDS_MAP = {
        "criminal": CRIMINAL_KEYWORDS,
        "civil": CIVIL_KEYWORDS,
        "rent": RENT_KEYWORDS,
        "labor": LABOR_KEYWORDS,
        "family": FAMILY_KEYWORDS,
        "defamation": DEFAMATION_KEYWORDS,
        "cyber": CYBER_KEYWORDS,
        "commercial": COMMERCIAL_KEYWORDS,
    }

    def __init__(self):
        self._nb_classifier = None

    def _get_nb_classifier(self):
        if self._nb_classifier is None:
            try:
                from src.classifier import CaseTypeClassifier
                self._nb_classifier = CaseTypeClassifier()
            except Exception:
                pass
        return self._nb_classifier

    def _score_domain(self, query: str, keywords: List[str]) -> int:
        score = 0
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', query):
                score += 1
        return score

    def _compute_keyword_scores(self, query_lower: str) -> dict:
        scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS_MAP.items():
            scores[domain] = self._score_domain(query_lower, keywords)
        return scores

    def _apply_negation(self, query_lower: str, scores: dict):
        for pattern in NEGATION_PATTERNS:
            if re.search(pattern, query_lower):
                for domain in ("criminal", "defamation", "cyber"):
                    scores[domain] = max(0, scores.get(domain, 0) - 3)

    def classify(self, query: str) -> Tuple[str, float, List[str]]:
        query_lower = query.lower()

        kw_scores = self._compute_keyword_scores(query_lower)
        self._apply_negation(query_lower, kw_scores)

        nb = self._get_nb_classifier()
        if nb:
            try:
                nb_result = nb.predict(query)
                nb_probs = nb_result["all_probabilities"]
            except Exception:
                nb_probs = {}
        else:
            nb_probs = {}

        domains = list(self.DOMAIN_KEYWORDS_MAP.keys())
        max_kw = max(kw_scores.values()) or 1
        combined = {}
        for d in domains:
            kw_norm = kw_scores.get(d, 0) / max_kw
            nb_prob = nb_probs.get(d, 0)
            combined[d] = 0.6 * kw_norm + 0.4 * nb_prob

        if kw_scores.get("criminal", 0) > 0 and combined["civil"] > 0:
            if max(combined["criminal"], combined["civil"]) >= 0.5:
                return ("both_criminal_civil", round(max(combined.values()), 2), ["criminal", "civil"])
        if kw_scores.get("rent", 0) > 0 and kw_scores.get("criminal", 0) > 0:
            return ("rent", 0.90, ["criminal", "civil"])
        if kw_scores.get("defamation", 0) > 0 and kw_scores.get("cyber", 0) > 0:
            return ("cyber_defamation", 0.85, ["defamation", "cyber", "criminal"])

        max_domain = max(combined, key=combined.get)
        max_score = combined[max_domain]
        confidence = min(0.95, max(0.3, max_score))

        secondary = [d for d in domains if d != max_domain and kw_scores.get(d, 0) >= 2]

        return (max_domain, round(confidence, 2), secondary)

    def get_response_type(self, primary_domain: str, secondary_domains: List[str]) -> str:
        if primary_domain in ("defamation", "cyber_defamation"):
            return "criminal_and_civil"
        if primary_domain == "both_criminal_civil":
            return "criminal_and_civil"
        if "criminal" in secondary_domains and primary_domain == "rent":
            return "criminal_and_civil"
        if primary_domain == "criminal":
            return "criminal_and_civil" if "civil" in secondary_domains else "criminal_only"
        if primary_domain == "civil":
            return "criminal_and_civil" if "criminal" in secondary_domains else "civil_only"
        return primary_domain
