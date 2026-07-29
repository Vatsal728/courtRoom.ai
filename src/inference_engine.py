import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_base import FolKnowledgeBase

class InferenceEngine:
    """Forward chaining inference over FOL rules"""
    
    def __init__(self):
        self.kb = FolKnowledgeBase()
        self.trace = []
    
    def build_working_memory(self, nlp_result: Dict, classifier_result: Dict) -> Set[str]:
        """Build initial working memory from NLP and classifier results"""
        working_memory = set()
        text_lower = nlp_result["original_text"].lower()
        
        # Add domain from classifier
        domain = classifier_result["primary_domain"]
        if domain == "consumer":
            working_memory.add("is_consumer")
        elif domain == "labour":
            working_memory.add("is_employee")
        elif domain == "rent":
            working_memory.add("is_tenant")
        
        # Check for defective product
        if any(word in text_lower for word in ["defective", "broken", "damaged", "faulty", "not working", "stopped working", "repair"]):
            working_memory.add("product_defective")
        
        # Check for payment
        if any(word in text_lower for word in ["paid", "money", "charge", "rupees", "rs", "cost", "price", "bought", "buy"]):
            working_memory.add("money_paid")
        
        # Check for delay
        if any(word in text_lower for word in ["delayed", "delay", "late", "waiting", "pending", "withheld", "not paid"]):
            working_memory.add("salary_delayed")
        
        # Check for time constraints
        # Default to within 2 years unless a long period like "3 years" is found
        if not ("3 years" in text_lower or "three years" in text_lower or "4 years" in text_lower or "5 years" in text_lower):
            working_memory.add("within_2_years")
            
        # Check for specific 3-month salary delay
        if any(term in text_lower for term in ["3 months", "three months", "4 months", "four months", "5 months", "90 days"]):
            working_memory.add("no_payment_3months")
            
        # Check for 30 days window
        if any(term in text_lower for term in ["days", "week"]) and not any(term in text_lower for term in ["30 days", "40 days", "50 days", "month"]):
            working_memory.add("within_30_days")
        
        # Check for online
        if any(word in text_lower for word in ["online", "website", "app", "ecommerce", "order", "internet"]):
            working_memory.add("online_purchase")
        
        # Check for eviction
        if any(word in text_lower for word in ["eviction", "evict", "throw", "remove", "vacate"]):
            working_memory.add("eviction_notice_invalid")
        
        # Check for illegality
        if any(word in text_lower for word in ["illegal", "wrong", "unlawful", "against"]):
            working_memory.add("eviction_notice_invalid")
        
        return working_memory
    
    def forward_chain(self, working_memory: Set[str], domain: str = None) -> Tuple[List[Dict], List[str]]:
        """Execute forward chaining to derive new facts"""
        self.trace = []
        new_facts_added = True
        iteration = 0
        
        while new_facts_added and iteration < 10:
            new_facts_added = False
            iteration += 1
            
            # Filter rules by domain if specified
            if domain:
                rules = self.kb.get_rule_by_domain(domain)
            else:
                rules = self.kb.rules
            
            for rule in rules:
                # Check if all conditions are met
                if all(cond in working_memory for cond in rule["conditions"]):
                    # Fire rule - add conclusion
                    if rule["conclusion"] not in working_memory:
                        working_memory.add(rule["conclusion"])
                        new_facts_added = True
                        
                        # Record trace
                        self.trace.append({
                            "rule_name": rule["name"],
                            "conditions_met": rule["conditions"],
                            "conclusion": rule["conclusion"],
                            "section": rule["section"],
                            "remedy": rule.get("remedy", ""),
                            "forum": rule.get("forum", ""),
                            "iteration": iteration
                        })
        
        return self.trace, list(working_memory)
    
    def explain(self, trace: List[Dict]) -> str:
        """Generate human-readable explanation of inference"""
        explanation = "Why this case applies to you:\n\n"
        
        for i, fired_rule in enumerate(trace, 1):
            explanation += f"{i}. {fired_rule['rule_name']}\n"
            explanation += f"   Section: {fired_rule['section']}\n"
            explanation += f"   Conclusion: {fired_rule['conclusion']}\n"
            if fired_rule.get('remedy'):
                explanation += f"   You can claim: {fired_rule['remedy']}\n"
            if fired_rule.get('forum'):
                explanation += f"   File at: {fired_rule['forum']}\n"
            explanation += "\n"
        
        return explanation

if __name__ == "__main__":
    from src.nlp_pipeline import NLPPipeline
    from src.classifier import CaseTypeClassifier
    
    # Setup
    nlp = NLPPipeline()
    classifier = CaseTypeClassifier()
    engine = InferenceEngine()
    
    # Test query
    query = (
        "Someone hacked my online bank account yesterday and transferred Rs 80,000 without my authorization. "
        "When I reached out to confront the suspect, they sent me threatening WhatsApp messages saying they "
        "will physically attack me and burn my house down if I report this. I have bank statements showing "
        "the unauthorized transfer and screenshots of the WhatsApp chat threats."
    )
    
    print(f"Query: {query}\n")
    
    # Process
    nlp_result = nlp.process_query(query)
    classifier_result = classifier.predict(query)
    
    # Infer
    working_memory = engine.build_working_memory(nlp_result, classifier_result)
    trace, final_facts = engine.forward_chain(working_memory, classifier_result["primary_domain"])
    
    print(f"Domain: {classifier_result['primary_domain']} ({classifier_result['confidence']:.2%})\n")
    print(engine.explain(trace))
