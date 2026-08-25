import re
from typing import List, Tuple

from src.domain_config import get_config, normalize_query


class DomainClassifier:
    """Classify query into legal domain with NaiveBayes + keyword ensemble.

    Fully config-driven: keyword lists, negation patterns, special rules,
    weights and NB class aliases all come from config/domain_config.json.
    """

    def __init__(self):
        self._nb_classifier = None

    def normalize_query(self, query: str) -> str:
        return normalize_query(query)

    def _get_nb_classifier(self):
        if self._nb_classifier is None:
            try:
                from src.classifier import CaseTypeClassifier
                self._nb_classifier = CaseTypeClassifier()
            except Exception:
                pass
        return self._nb_classifier

    @staticmethod
    def _score_domain(query: str, keywords: List[str]) -> int:
        score = 0
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', query):
                score += 1
        return score

    def _compute_keyword_scores(self, query_lower: str) -> dict:
        cfg = get_config()
        scores = {}
        for domain, spec in cfg.get("domains", {}).items():
            scores[domain] = self._score_domain(query_lower, spec.get("keywords", []))
        return scores

    def _apply_negation(self, query_lower: str, scores: dict):
        cfg = get_config()
        patterns = cfg.get("negation_patterns", [])
        domains = cfg.get("negation_domains", [])
        for pattern in patterns:
            if re.search(pattern, query_lower):
                for domain in domains:
                    scores[domain] = max(0, scores.get(domain, 0) - 3)

    def _nb_probabilities(self, query: str) -> dict:
        nb = self._get_nb_classifier()
        if nb:
            try:
                return nb.predict(query)["all_probabilities"]
            except Exception:
                pass
        return {}

    def classify(self, query: str) -> Tuple[str, float, List[str]]:
        cfg = get_config()
        query = self.normalize_query(query)
        query_lower = query.lower()

        kw_scores = self._compute_keyword_scores(query_lower)
        self._apply_negation(query_lower, kw_scores)

        nb_probs = self._nb_probabilities(query)
        nb_map = cfg.get("nb_class_map", {})

        domains = list(cfg.get("domains", {}).keys())
        max_kw = max(kw_scores.values()) or 1
        combined = {}
        for d in domains:
            kw_norm = kw_scores.get(d, 0) / max_kw
            nb_prob = nb_probs.get(nb_map.get(d, d), 0)
            combined[d] = (
                cfg["classifier"]["keyword_weight"] * kw_norm
                + cfg["classifier"]["nb_weight"] * nb_prob
            )

        for rule in cfg.get("rules", []):
            min_kw = rule.get("domains", {})
            if all(kw_scores.get(d, 0) >= m for d, m in min_kw.items()):
                if "min_combined" in rule and max(combined.values()) < rule["min_combined"]:
                    continue
                secondary = rule.get("secondary", [])
                if "fixed_confidence" in rule:
                    return (rule["result"], float(rule["fixed_confidence"]), secondary)
                return (rule["result"], round(max(combined.values()), 2), secondary)

        # If the query hits no domain keywords at all, don't let a low-probability
        # NaiveBayes guess dominate (e.g. "bought a defective phone" -> "cyber").
        if max(kw_scores.values()) == 0:
            max_combined = max(combined.values())
            threshold = cfg["classifier"]["no_keyword_max_combined"]
            if max_combined < threshold:
                return (
                    cfg["classifier"]["no_keyword_fallback_domain"],
                    round(max(0.3, max_combined), 2),
                    [],
                )

        max_domain = max(combined, key=combined.get)
        max_score = combined[max_domain]
        confidence = min(cfg["classifier"]["max_confidence"], max(0.3, max_score))

        secondary_min = cfg["classifier"]["secondary_min_keywords"]
        secondary = [d for d in domains if d != max_domain and kw_scores.get(d, 0) >= secondary_min]

        return (max_domain, round(confidence, 2), secondary)

    def get_response_type(self, primary_domain: str, secondary_domains: List[str]) -> str:
        cfg = get_config().get("response_types", {})
        for dom, rule in cfg.get("secondary_rules", {}).items():
            if primary_domain == dom and any(s in rule.get("secondary_in", []) for s in secondary_domains):
                return rule["result"]
        for result, domains in cfg.get("primary_rules", {}).items():
            if primary_domain in domains:
                return result
        return primary_domain
