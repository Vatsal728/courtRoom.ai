"""
compensation_engine.py - Deterministic compensation estimator.

Every figure produced here is computed from config/compensation_rules.json and
the extracted facts (amounts, durations, reach/income bands). The LLM never
generates numbers; it only supplies the facts. Output is a range with basis
and disclaimer.
"""
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv(override=True)


class CompensationEngine:
    def __init__(self, rules_path: str = None):
        path = rules_path or str(
            Path(__file__).resolve().parent.parent / "config" / "compensation_rules.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            self._rules = json.load(f)
        self._lock = threading.Lock()

    @property
    def currency(self) -> str:
        return self._rules.get("currency", "INR")

    @property
    def disclaimer(self) -> str:
        return self._rules.get(
            "disclaimer",
            "Compensation figures are estimates and not legal advice.",
        )

    def available_domains(self):
        return sorted(self._rules.get("rules", {}).keys())

    def _rule(self, domain: str) -> Optional[Dict[str, Any]]:
        rules = self._rules.get("rules", {})
        for key, rule in rules.items():
            if domain == key:
                return rule
            if domain in rule.get("aliases", []):
                return rule
        return None

    def estimate(self, domain: str, facts: Dict[str, Any]) -> Dict[str, Any]:
        """Return a deterministic compensation range for `domain` from `facts`."""
        rule = self._rule(domain) or self._rule("civil")
        if not rule:
            return {
                "domain": domain,
                "min_amount": 0,
                "max_amount": 0,
                "currency": self.currency,
                "basis": "No rule configured for this domain.",
                "notes": [],
                "disclaimer": self.disclaimer,
            }
        params = rule.get("params", {})
        formula = rule.get("formula", "band")
        handler = getattr(self, f"_f_{formula}", self._f_band)
        lo, hi, basis = handler(facts, params)
        lo, hi = max(0, lo), max(lo, hi)
        return {
            "domain": domain,
            "min_amount": round(lo, 2),
            "max_amount": round(hi, 2),
            "currency": self.currency,
            "basis": basis,
            "notes": rule.get("notes", []),
            "disclaimer": self.disclaimer,
        }

    # ── Fact helpers ────────────────────────────────────────────────────
    @staticmethod
    def _f(facts: Dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(facts.get(key) or default)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _years(facts: Dict[str, Any]) -> float:
        y = CompensationEngine._f(facts, "years")
        if y > 0:
            return y
        m = CompensationEngine._f(facts, "months")
        return m / 12.0

    # ── Formulas ────────────────────────────────────────────────────────
    def _f_deposit_refund(self, facts, p) -> tuple:
        deposit = self._f(facts, "deposit_amount")
        years = self._years(facts)
        refund = deposit * (p.get("refund_rate_pct", 100) / 100.0)
        interest = refund * p.get("interest_rate_annual", 0.06) * years
        return (refund, refund + interest,
                f"Deposit refund of Rs {deposit:,.0f} plus ~{p.get('interest_rate_annual', 0.06) * 100:.0f}% p.a. interest on the delayed refund")

    def _f_wages_and_compensation(self, facts, p) -> tuple:
        unpaid = self._f(facts, "unpaid_amount")
        if unpaid <= 0:
            monthly = self._f(facts, "monthly_wage")
            months = self._f(facts, "months")
            unpaid = monthly * months
        years = self._years(facts)
        interest = unpaid * p.get("wage_interest_annual", 0.12) * max(years, 0.5)
        yos = self._f(facts, "years_service")
        daily = self._f(facts, "avg_daily_wage")
        retrenchment = 0.0
        if yos > 0 and daily > 0:
            retrenchment = daily * p.get("retrenchment_days_per_year", 15) * yos
        gratuity = 0.0
        if yos >= 5 and daily > 0:
            gratuity = min(daily * p.get("gratuity_days_per_year", 15) * yos,
                           p.get("gratuity_cap", 3500000))
        lo = unpaid
        hi = unpaid + interest + retrenchment + gratuity
        return (lo, hi,
                f"Unpaid wages Rs {unpaid:,.0f} + ~{p.get('wage_interest_annual', 0.12) * 100:.0f}% p.a. interest"
                + (f" + retrenchment ~Rs {retrenchment:,.0f}" if retrenchment else "")
                + (f" + gratuity ~Rs {gratuity:,.0f}" if gratuity else ""))

    def _f_loss_multiplier(self, facts, p) -> tuple:
        loss = self._f(facts, "loss_amount") or self._f(facts, "amount")
        if loss <= 0:
            return (0.0, 0.0, "Loss amount not provided; no estimate possible.")
        multiplier = 1.0
        for tier in p.get("tiers", []):
            if "max_loss" in tier and loss > tier["max_loss"]:
                continue
            multiplier = tier.get("multiplier", 1.0)
            break
        max_mult = p.get("max_multiplier", 2.0)
        lo = loss * multiplier
        hi = loss * min(multiplier + 0.5, max_mult)
        return (lo, hi,
                f"Loss of Rs {loss:,.0f} x tier multiplier {multiplier:g}, capped at {max_mult:g}x")

    def _f_band(self, facts, p) -> tuple:
        reach = str(facts.get("reach") or facts.get("income") or "").lower()
        chosen = None
        for band in p.get("bands", []):
            if "reach" in band and band["reach"] == reach:
                chosen = band
                break
            if "income_max" in band:
                try:
                    income = float(facts.get("income") or 0)
                    if income <= band["income_max"]:
                        chosen = band
                        break
                except (TypeError, ValueError):
                    continue
        if chosen is None:
            chosen = p.get("bands", [{}])[-1] if p.get("bands") else {"min": 0, "max": 0}
        return (chosen.get("min", 0), chosen.get("max", 0),
                f"Band-based estimate for reach/category '{reach or 'default'}'")

    def _f_criminal_general(self, facts, p) -> tuple:
        stolen = self._f(facts, "stolen_value") or self._f(facts, "loss_amount")
        cheated = self._f(facts, "cheated_amount") or self._f(facts, "amount")
        medical = self._f(facts, "medical_bills")
        years = self._years(facts)
        parts = []
        lo, hi = 0.0, 0.0
        if stolen > 0:
            lo += stolen
            hi += stolen * p.get("stolen_value_factor", 1.0)
            parts.append(f"value of stolen property Rs {stolen:,.0f}")
        if cheated > 0:
            interest = cheated * p.get("cheating_interest_annual", 0.12) * max(years, 0.5)
            lo += cheated
            hi += cheated + interest
            parts.append(f"amount cheated Rs {cheated:,.0f} + interest ~{p.get('cheating_interest_annual', 0.12) * 100:.0f}% p.a.")
        if medical > 0:
            pain = min(medical * p.get("assault_medical_factor", 2.0), p.get("assault_pain_cap", 250000))
            lo += medical
            hi += medical + pain
            parts.append(f"medical bills Rs {medical:,.0f} + pain-and-suffering up to Rs {pain:,.0f}")
        basis = "; ".join(parts) if parts else "No monetary loss provided; damages depend on the injury proved."
        return (lo, hi, basis)

    def _f_contract_loss(self, facts, p) -> tuple:
        loss = self._f(facts, "loss_amount") or self._f(facts, "amount")
        years = self._years(facts)
        interest = loss * p.get("interest_annual", 0.18) * max(years, 0.5)
        return (loss, loss + interest,
                f"Direct loss Rs {loss:,.0f} + interest ~{p.get('interest_annual', 0.18) * 100:.0f}% p.a.")


_engine = None
_engine_lock = threading.Lock()


def get_compensation_engine() -> CompensationEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = CompensationEngine()
    return _engine


if __name__ == "__main__":
    eng = get_compensation_engine()
    tests = [
        ("rent", {"deposit_amount": 50000, "months": 3}),
        ("consumer", {"loss_amount": 50000}),
        ("labor", {"monthly_wage": 15000, "months": 2, "years_service": 3, "avg_daily_wage": 600}),
        ("criminal", {"stolen_value": 20000}),
        ("criminal", {"medical_bills": 15000}),
        ("family", {"income": 25000}),
        ("defamation", {"reach": "public"}),
    ]
    for d, f in tests:
        r = eng.estimate(d, f)
        print(f"{d:10} -> Rs {r['min_amount']:,.0f} - Rs {r['max_amount']:,.0f} | {r['basis'][:80]}")
