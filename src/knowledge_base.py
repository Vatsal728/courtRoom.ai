from typing import Dict, List, Set

class FolKnowledgeBase:
    """First Order Logic rule base for Indian law"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[Dict]:
        """Define all FOL rules for legal domains"""
        rules = [
            # Consumer Protection Act rules
            {
                "name": "consumer_complaint_eligible",
                "domain": "consumer",
                "conditions": ["is_consumer", "product_defective", "money_paid", "within_2_years"],
                "conclusion": "can_file_consumer_complaint",
                "section": "Consumer Protection Act 2019, Section 35",
                "remedy": "Refund, replacement, or compensation",
                "forum": "District Consumer Commission"
            },
            {
                "name": "online_purchase_eligible",
                "domain": "consumer",
                "conditions": ["online_purchase", "not_received", "money_paid", "within_30_days"],
                "conclusion": "can_file_consumer_complaint",
                "section": "Consumer Protection Act 2019, Section 35",
                "remedy": "Full refund or replacement",
                "forum": "District Consumer Commission"
            },
            
            # Labour Law rules
            {
                "name": "salary_delay_eligible",
                "domain": "labour",
                "conditions": ["is_employee", "salary_delayed", "worked_days_present", "no_payment_3months"],
                "conclusion": "can_file_labour_complaint",
                "section": "Payment of Wages Act 1936, Section 15",
                "remedy": "Full salary payment + interest",
                "forum": "Labour Commissioner"
            },
            {
                "name": "illegal_termination",
                "domain": "labour",
                "conditions": ["is_employee", "terminated_without_notice", "no_compensation"],
                "conclusion": "can_file_termination_complaint",
                "section": "Payment of Wages Act 1936, Section 9",
                "remedy": "Compensation + back wages",
                "forum": "Labour Commissioner"
            },
            {
                "name": "below_minimum_wage",
                "domain": "labour",
                "conditions": ["is_employee", "paid_below_minimum", "worked_days_present"],
                "conclusion": "can_file_wage_complaint",
                "section": "Minimum Wages Act 1948, Section 24",
                "remedy": "Difference in wages + interest",
                "forum": "Labour Commissioner"
            },
            
            # RTI rules
            {
                "name": "rti_eligible",
                "domain": "rti",
                "conditions": ["requested_govt_info", "within_reasonable_time", "no_exemption"],
                "conclusion": "can_file_rti",
                "section": "RTI Act 2005, Section 6",
                "remedy": "Government must provide information",
                "deadline": "30 days from request"
            },
            {
                "name": "rti_appeal",
                "domain": "rti",
                "conditions": ["rti_rejected", "within_30_days"],
                "conclusion": "can_file_rti_appeal",
                "section": "RTI Act 2005, Section 19",
                "remedy": "First appeal to senior PIO",
                "deadline": "30 days from rejection"
            },
            
            # Rent Act rules
            {
                "name": "illegal_eviction",
                "domain": "rent",
                "conditions": ["is_tenant", "eviction_notice_invalid", "no_court_order"],
                "conclusion": "can_challenge_eviction",
                "section": "Gujarat Rent Control Act 1999, Section 12",
                "remedy": "Eviction order quashed, can continue tenancy",
                "forum": "District Court"
            },
            {
                "name": "deposit_refund",
                "domain": "rent",
                "conditions": ["tenant_vacated", "no_damage", "landlord_refusing_deposit"],
                "conclusion": "can_demand_deposit_refund",
                "section": "Gujarat Rent Control Act 1999, Section 11",
                "remedy": "Deposit refund with interest",
                "forum": "District Court"
            },
            
            # Cyber Law rules
            {
                "name": "online_fraud",
                "domain": "cyber",
                "conditions": ["money_transferred", "fraudulent_website", "evidence_present"],
                "conclusion": "can_file_cyber_complaint",
                "section": "IT Act 2000, Section 66",
                "remedy": "Criminal action + recovery of money",
                "forum": "Cybercrime Police"
            },
            {
                "name": "account_hacking",
                "domain": "cyber",
                "conditions": ["account_accessed", "unauthorized_transaction", "evidence"],
                "conclusion": "can_file_hacking_complaint",
                "section": "IT Act 2000, Section 66",
                "remedy": "Police investigation, recovery",
                "forum": "Cybercrime Police"
            }
        ]
        
        return rules
    
    def get_applicable_rules(self, working_memory: Set[str]) -> List[Dict]:
        """Forward chaining: find all rules that match current facts"""
        applicable = []
        
        for rule in self.rules:
            # Check if all conditions are in working memory
            if all(cond in working_memory for cond in rule["conditions"]):
                applicable.append(rule)
        
        return applicable
    
    def get_rule_by_domain(self, domain: str) -> List[Dict]:
        """Get all rules for a specific domain"""
        return [r for r in self.rules if r["domain"] == domain]

if __name__ == "__main__":
    kb = FolKnowledgeBase()
    
    # Test: Consumer complaint case
    working_memory = {
        "is_employee", "salary_delayed", "worked_days_present", "no_payment_3months"
    }
    
    applicable = kb.get_applicable_rules(working_memory)
    print(f"For working memory: {working_memory}")
    print(f"\nApplicable rules ({len(applicable)}):")
    for rule in applicable:
        print(f"\n  Rule: {rule['name']}")
        print(f"  Conclusion: {rule['conclusion']}")
        print(f"  Section: {rule['section']}")
        print(f"  Remedy: {rule['remedy']}")
