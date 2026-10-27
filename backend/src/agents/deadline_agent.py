import sys
import io
from datetime import datetime, timedelta
from typing import Dict

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class DeadlineTrackerAgent:
    """Track statutory limitation periods and filing deadlines under Indian law."""

    def __init__(self):
        self.deadlines = {
            "consumer": {
                "limitation_period": 2,
                "unit": "years",
                "description": "File consumer complaint within 2 years from cause of action (Section 69, Consumer Protection Act 2019)",
            },
            "labour_salary": {
                "limitation_period": 3,
                "unit": "years",
                "description": "File wage recovery claim within 3 years from due date (Section 45(4), Code on Wages 2019; 12 months under Payment of Wages Act 1936)",
            },
            "rti_appeal": {
                "limitation_period": 30,
                "unit": "days",
                "description": "File RTI First Appeal within 30 days of PIO rejection or expiry of 30-day response window (Section 19(1), RTI Act 2005)",
            },
            "rti_second_appeal": {
                "limitation_period": 90,
                "unit": "days",
                "description": "File RTI Second Appeal before Information Commission within 90 days of First Appellate order (Section 19(3), RTI Act 2005)",
            },
            "rent_eviction": {
                "limitation_period": 30,
                "unit": "days",
                "description": "Respond to statutory tenancy/eviction notice or challenge unlawful eviction within 30 days",
            },
            "criminal_fir": {
                "limitation_period": 3,
                "unit": "years",
                "description": "Take cognizance within 3 years for offences punishable up to 3 years (Section 514 BNSS 2023; no limitation for grave offences)",
            },
            "civil_recovery": {
                "limitation_period": 3,
                "unit": "years",
                "description": "File civil suit for money recovery or breach of contract within 3 years (Articles 19/55, Limitation Act 1963)",
            },
            "cheque_bounce": {
                "limitation_period": 30,
                "unit": "days",
                "description": "Issue statutory demand notice within 30 days of cheque dishonour memo (Section 138(b), Negotiable Instruments Act 1881)",
            },
        }

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        s = (date_str or "").strip()
        for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d.%m.%Y"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unsupported date format: '{date_str}'. Expected DD-MM-YYYY or YYYY-MM-DD.")

    def calculate_deadline(self, case_type: str, incident_date: str) -> Dict:
        """Calculate statutory deadline for filing."""
        try:
            incident = self._parse_date(incident_date)
            today = datetime.now()

            if case_type not in self.deadlines:
                return {"error": f"Unknown case type: {case_type}"}

            deadline_info = self.deadlines[case_type]
            period = deadline_info["limitation_period"]
            unit = deadline_info.get("unit", "years")

            if isinstance(period, int):
                if unit == "days":
                    deadline = incident + timedelta(days=period)
                elif unit == "months":
                    deadline = incident + timedelta(days=30 * period)
                else:
                    deadline = incident + timedelta(days=365 * period)
            else:
                deadline = None

            days_remaining = (deadline - today).days if deadline else None

            if days_remaining is None:
                status = "✅ OK"
            elif days_remaining < 0:
                status = "❌ EXPIRED"
            elif days_remaining < 30:
                status = "⚠️ URGENT"
            else:
                status = "✅ OK"

            return {
                "case_type": case_type,
                "incident_date": incident.strftime("%d-%m-%Y"),
                "deadline_date": deadline.strftime("%d-%m-%Y") if deadline else "See description",
                "days_remaining": days_remaining,
                "status": status,
                "description": deadline_info["description"],
            }

        except Exception as e:
            return {"error": str(e)}

    def print_deadline(self, case_type: str, incident_date: str) -> str:
        """Print formatted deadline reminder."""
        result = self.calculate_deadline(case_type, incident_date)

        if "error" in result:
            return f"❌ Error: {result['error']}"

        return f"""
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
