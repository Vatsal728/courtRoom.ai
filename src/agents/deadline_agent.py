import sys
import io
from datetime import datetime, timedelta
from typing import Dict

# Set console output encoding to UTF-8 to prevent UnicodeEncodeError on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class DeadlineTrackerAgent:
    """Track limitation periods and filing deadlines"""
    
    def __init__(self):
        self.deadlines = {
            "consumer": {
                "limitation_period": 2,  # years
                "description": "File consumer complaint within 2 years from deficiency"
            },
            "labour_salary": {
                "limitation_period": 3,  # months from delay
                "description": "File wage complaint within 3 months of salary delay"
            },
            "rti_appeal": {
                "limitation_period": 30,  # days
                "description": "File RTI first appeal within 30 days of PIO rejection"
            },
            "rti_second_appeal": {
                "limitation_period": 90,  # days
                "description": "File RTI second appeal within 90 days of first appeal rejection"
            },
            "rent_eviction": {
                "limitation_period": "immediate",
                "description": "Challenge eviction notice within 30 days in district court"
            },
            "criminal_fir": {
                "limitation_period": 3,  # years for most crimes
                "description": "File criminal complaint within 3 years (some crimes: no limit)"
            }
        }
    
    def calculate_deadline(self, case_type: str, incident_date: str) -> Dict:
        """Calculate deadline for filing"""
        try:
            incident = datetime.strptime(incident_date, "%d-%m-%Y")
            today = datetime.now()
            
            if case_type not in self.deadlines:
                return {"error": "Unknown case type"}
            
            deadline_info = self.deadlines[case_type]
            period = deadline_info["limitation_period"]
            
            if isinstance(period, int):
                # Years or days
                if case_type in ["rti_appeal", "rti_second_appeal"]:
                    deadline = incident + timedelta(days=period)
                else:
                    # Treat others as years
                    deadline = incident + timedelta(days=365 * period)
            else:
                deadline = None
            
            days_remaining = (deadline - today).days if deadline else None
            
            if days_remaining is None:
                status = "✅ OK"  # Or "N/A" for immediate/no limitation cases
            elif days_remaining < 0:
                status = "❌ EXPIRED"
            elif days_remaining < 30:
                status = "⚠️ URGENT"
            else:
                status = "✅ OK"
            
            return {
                "case_type": case_type,
                "incident_date": incident_date,
                "deadline_date": deadline.strftime("%d-%m-%Y") if deadline else "See description",
                "days_remaining": days_remaining,
                "status": status,
                "description": deadline_info["description"]
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def print_deadline(self, case_type: str, incident_date: str) -> str:
        """Print formatted deadline reminder"""
        result = self.calculate_deadline(case_type, incident_date)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        output = f"""
⏰ DEADLINE TRACKER
{'='*50}
Case Type: {result['case_type'].upper()}
Incident Date: {result['incident_date']}
Filing Deadline: {result['deadline_date']}
Days Remaining: {result['days_remaining']}
Status: {result['status']}

Description: {result['description']}
{'='*50}
"""
        return output

if __name__ == "__main__":
    agent = DeadlineTrackerAgent()
    
    # Test deadline
    print(agent.print_deadline("consumer", "15-07-2024"))
    print(agent.print_deadline("rti_appeal", "10-07-2024"))
