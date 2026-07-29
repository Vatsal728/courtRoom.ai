"""
domain_classifier.py - Improved domain detection
Detects: Criminal vs Civil vs Labor vs Property vs Family vs Commercial
"""

import re
from typing import Tuple, List

class DomainClassifier:
    """Classify query into legal domain with high accuracy"""
    
    # Keywords for each domain
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
    
    CRIMINAL_INTIMIDATION = [
        "threat", "intimidation", "threatening", "terrorize", "scare",
        "blackmail", "extortion", "coercion", "force", "compel"
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
    
    def classify(self, query: str) -> Tuple[str, float, List[str]]:
        """
        Classify query into domain
        
        Returns:
            (primary_domain, confidence_score, secondary_domains)
        """
        
        query_lower = query.lower()
        scores = {}
        
        # Score each domain
        scores['criminal'] = self._score_domain(query_lower, self.CRIMINAL_KEYWORDS)
        scores['civil'] = self._score_domain(query_lower, self.CIVIL_KEYWORDS)
        scores['rent'] = self._score_domain(query_lower, self.RENT_KEYWORDS)
        scores['labor'] = self._score_domain(query_lower, self.LABOR_KEYWORDS)
        scores['family'] = self._score_domain(query_lower, self.FAMILY_KEYWORDS)
        scores['defamation'] = self._score_domain(query_lower, self.DEFAMATION_KEYWORDS)
        scores['cyber'] = self._score_domain(query_lower, self.CYBER_KEYWORDS)
        scores['commercial'] = self._score_domain(query_lower, self.COMMERCIAL_KEYWORDS)
        
        # Special rules for combined domains
        if scores['criminal'] > 0 and scores['civil'] > 0:
            # Could be criminal + civil
            if max(scores['criminal'], scores['civil']) >= 3:
                return ('both_criminal_civil', 0.85, ['criminal', 'civil'])
        
        if scores['rent'] > 0:
            if scores['criminal'] > 0:
                return ('rent', 0.90, ['criminal', 'civil'])
        
        if scores['defamation'] > 0 and scores['cyber'] > 0:
            return ('cyber_defamation', 0.85, ['defamation', 'cyber', 'criminal'])
        
        # Find dominant domain
        max_domain = max(scores, key=scores.get)
        max_score = scores[max_domain]
        
        # Calculate confidence (0-1)
        if max_score == 0:
            confidence = 0.3  # Very low confidence - unclear
        else:
            confidence = min(0.9, max_score / 10)
        
        # Find secondary domains
        secondary = [d for d, s in scores.items() if d != max_domain and s > 2]
        
        return (max_domain, confidence, secondary)
    
    def _score_domain(self, query: str, keywords: List[str]) -> int:
        """Score domain based on keyword matches"""
        score = 0
        for keyword in keywords:
            if keyword in query:
                score += 1
        return score
    
    def get_response_type(self, primary_domain: str, secondary_domains: List[str]) -> str:
        """
        Determine response type:
        - criminal_only
        - civil_only
        - criminal_and_civil
        - labor
        - family
        - commercial
        """
        
        if primary_domain in ['defamation', 'cyber_defamation']:
            return 'criminal_and_civil'
        
        if primary_domain == 'both_criminal_civil':
            return 'criminal_and_civil'
        
        if 'criminal' in secondary_domains and primary_domain == 'rent':
            return 'criminal_and_civil'
        
        if primary_domain == 'criminal':
            if 'civil' in secondary_domains:
                return 'criminal_and_civil'
            return 'criminal_only'
        
        if primary_domain == 'civil':
            if 'criminal' in secondary_domains:
                return 'criminal_and_civil'
            return 'civil_only'
        
        return primary_domain
