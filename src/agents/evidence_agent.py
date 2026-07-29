import sys
import io
from typing import List, Dict

# Set console output encoding to UTF-8 to prevent UnicodeEncodeError on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class EvidenceChecklistAgent:
    """Generate domain-specific evidence checklist"""
    
    def __init__(self):
        self.checklists = {
            "consumer": [
                "Purchase receipt / invoice",
                "Product packaging and photos",
                "Warranty card / guarantee certificate",
                "Communication with seller (emails, messages)",
                "Screenshots of product listing",
                "Photos of defect / damage",
                "Proof of payment (bank statement, screenshot)",
                "Complaint emails / chat transcripts"
            ],
            "labour": [
                "Appointment letter",
                "All salary slips (last 12 months)",
                "Bank statements showing salary deposits/non-deposits",
                "Emails from employer",
                "WhatsApp messages / communications",
                "Employee ID / badge",
                "Performance reviews",
                "Termination letter (if fired)",
                "Company attendance records",
                "Proof of work done"
            ],
            "rent": [
                "Lease agreement / rental contract",
                "Security deposit receipt",
                "Rent payment receipts / bank transfers",
                "Photographs of property condition (entry + exit)",
                "Maintenance complaints (emails/letters)",
                "Eviction notice letter",
                "Utility bills in your name",
                "Correspondence with landlord"
            ],
            "rti": [
                "Your identity proof (Aadhaar, PAN, etc)",
                "Proof of residency",
                "Description of information requested",
                "Form A (RTI application form)",
                "Proof of fee payment (Rs 10)"
            ],
            "criminal": [
                "FIR copy (from police station)",
                "Medical report (if injury)",
                "Photographs of injury / damage",
                "Witness names and contacts",
                "Police case number",
                "Your written statement",
                "Evidence (recovered items, etc)",
                "Video footage (if available)"
            ],
            "cyber": [
                "Screenshots of fraudulent website / email",
                "Email headers showing source",
                "Banking / transaction records",
                "Police complaint (online or physical)",
                "Communication with bank / service provider",
                "Proof of amount lost",
                "Hacking incident details and timeline",
                "Device security reports"
            ]
        }
    
    def get_checklist(self, domain: str) -> Dict:
        """Get evidence checklist for domain"""
        if domain not in self.checklists:
            return {"domain": "unknown", "items": []}
        
        return {
            "domain": domain,
            "items": self.checklists[domain],
            "total_items": len(self.checklists[domain]),
            "instruction": f"Gather ALL of these documents before filing your {domain} complaint. Missing documents weaken your case."
        }
    
    def print_checklist(self, domain: str) -> str:
        """Generate formatted checklist"""
        checklist = self.get_checklist(domain)
        
        output = f"\n📋 Evidence Checklist for {domain.upper()} Case\n"
        output += "=" * 50 + "\n"
        output += f"Total items to gather: {checklist['total_items']}\n\n"
        
        for i, item in enumerate(checklist['items'], 1):
            output += f"☐ {i}. {item}\n"
        
        output += "\n" + checklist['instruction'] + "\n"
        return output

if __name__ == "__main__":
    agent = EvidenceChecklistAgent()
    
    for domain in ["consumer", "labour", "rent"]:
        print(agent.print_checklist(domain))
