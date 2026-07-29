import heapq
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

@dataclass
class LegalState:
    """A state in the legal case progression"""
    name: str
    description: str
    depth: int = 0
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name

class LegalStateSpace:
    """Define the legal case progression state space"""
    
    def __init__(self):
        self.states = self._define_states()
        self.edges = self._define_transitions()
    
    def _define_states(self) -> Dict[str, LegalState]:
        """Define all possible states in case progression"""
        return {
            "has_problem": LegalState("has_problem", "User identifies legal issue", 0),
            "gathered_evidence": LegalState("gathered_evidence", "Collected documents and proof", 1),
            "sent_notice": LegalState("sent_notice", "Sent legal notice to opposite party", 2),
            "waiting_response": LegalState("waiting_response", "Waiting for their response (15 days)", 3),
            "filed_complaint": LegalState("filed_complaint", "Filed at appropriate forum", 4),
            "hearing_scheduled": LegalState("hearing_scheduled", "First hearing date fixed", 5),
            "evidence_presented": LegalState("evidence_presented", "Presented evidence in court", 6),
            "case_resolved": LegalState("case_resolved", "Favorable order received", 7),
        }
    
    def _define_transitions(self) -> Dict[str, List[Tuple[str, int]]]:
        """Define edges: from_state -> [(to_state, cost)]"""
        return {
            "has_problem": [
                ("gathered_evidence", 3),  # Cost: days to gather evidence
                ("sent_notice", 5)
            ],
            "gathered_evidence": [
                ("sent_notice", 2),
                ("filed_complaint", 5)
            ],
            "sent_notice": [
                ("waiting_response", 15),
                ("filed_complaint", 10)
            ],
            "waiting_response": [
                ("filed_complaint", 5),
                ("case_resolved", 20)
            ],
            "filed_complaint": [
                ("hearing_scheduled", 30),
            ],
            "hearing_scheduled": [
                ("evidence_presented", 30),
            ],
            "evidence_presented": [
                ("case_resolved", 60),
            ],
            "case_resolved": []
        }
    
    def heuristic(self, current: str, evidence_strength: float) -> float:
        """Heuristic: estimated days to reach case_resolved"""
        # Better evidence → lower heuristic (faster resolution)
        base_days = max(0, 180 - (evidence_strength * 100))
        return base_days

class AStarLegalPathFinder:
    """A* search to find optimal legal pathway"""
    
    def __init__(self, evidence_strength: float = 0.5):
        self.space = LegalStateSpace()
        self.evidence_strength = evidence_strength
        self.path = []
    
    def search(self) -> List[str]:
        """A* search from has_problem to case_resolved"""
        start = "has_problem"
        goal = "case_resolved"
        
        # Priority queue: (f_score, counter, state)
        open_set = [(0, 0, start)]
        g_score = {start: 0}
        came_from = {}
        counter = 1
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = [goal]
                while current in came_from:
                    current = came_from[current]
                    path.insert(0, current)
                self.path = path
                return path
            
            # Get neighbors
            if current in self.space.edges:
                for neighbor, cost in self.space.edges[current]:
                    tentative_g = g_score[current] + cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        
                        h = self.space.heuristic(neighbor, self.evidence_strength)
                        f = tentative_g + h
                        
                        heapq.heappush(open_set, (f, counter, neighbor))
                        counter += 1
        
        return []  # No path found

class GoalStackPlanner:
    """Decompose A* path into goal stack (hierarchical to-do list)"""
    
    def __init__(self):
        self.goal_stack = []
    
    def plan(self, a_star_path: List[str]) -> List[Dict]:
        """Convert A* path to actionable goal stack"""
        goals = []
        
        for i, state in enumerate(a_star_path):
            if state == "has_problem":
                goals.append({
                    "order": i + 1,
                    "priority": "CRITICAL",
                    "goal": "Understand your legal position",
                    "actions": [
                        "Document all facts of the case",
                        "Gather all evidence (bills, messages, receipts)",
                        "Note dates and people involved"
                    ],
                    "deadline": "Today"
                })
            
            elif state == "gathered_evidence":
                goals.append({
                    "order": i + 1,
                    "priority": "CRITICAL",
                    "goal": "Collect all supporting documents",
                    "actions": [
                        "Gather bills, invoices, receipts",
                        "Screenshot all messages/emails",
                        "Get witness names and contact",
                        "Make copies of all documents"
                    ],
                    "deadline": "Within 3 days"
                })
            
            elif state == "sent_notice":
                goals.append({
                    "order": i + 1,
                    "priority": "HIGH",
                    "goal": "Send legal notice",
                    "actions": [
                        "Draft notice citing relevant law section",
                        "Demand specific compensation amount",
                        "Set 15-day response deadline",
                        "Send via registered post with acknowledgment"
                    ],
                    "deadline": "Within 1 week"
                })
            
            elif state == "waiting_response":
                goals.append({
                    "order": i + 1,
                    "priority": "MEDIUM",
                    "goal": "Wait for their response",
                    "actions": [
                        "Keep acknowledgment safe",
                        "Note the 15-day deadline",
                        "If they respond, analyze their position"
                    ],
                    "deadline": "15 days from notice"
                })
            
            elif state == "filed_complaint":
                goals.append({
                    "order": i + 1,
                    "priority": "CRITICAL",
                    "goal": "File complaint at appropriate forum",
                    "actions": [
                        "Get form from forum office",
                        "Fill with all details",
                        "Attach proof and legal notice copy",
                        "Pay filing fee",
                        "Submit and get receipt"
                    ],
                    "deadline": "Before limitation period expires"
                })
            
            elif state == "hearing_scheduled":
                goals.append({
                    "order": i + 1,
                    "priority": "HIGH",
                    "goal": "Prepare for first hearing",
                    "actions": [
                        "Get hearing date slip",
                        "Prepare opening statement",
                        "Arrange witnesses if needed",
                        "Bring all documents in folder"
                    ],
                    "deadline": "3 days before hearing"
                })
            
            elif state == "evidence_presented":
                goals.append({
                    "order": i + 1,
                    "priority": "CRITICAL",
                    "goal": "Present evidence effectively",
                    "actions": [
                        "Speak clearly and calmly",
                        "Stick to facts, no emotions",
                        "Answer only what is asked",
                        "Let documents speak for themselves"
                    ],
                    "deadline": "On hearing date"
                })
            
            elif state == "case_resolved":
                goals.append({
                    "order": i + 1,
                    "priority": "HIGH",
                    "goal": "Get favorable order",
                    "actions": [
                        "Get certified copy of order",
                        "Collect compensation if awarded",
                        "File appeal if unfavorable",
                        "Keep all documents safe"
                    ],
                    "deadline": "At case conclusion"
                })
        
        self.goal_stack = goals
        return goals

if __name__ == "__main__":
    # Test with high evidence strength
    print("=== A* Legal Pathway ===\n")
    
    finder = AStarLegalPathFinder(evidence_strength=0.8)
    path = finder.search()
    
    print(f"Optimal legal pathway ({len(path)} steps):")
    for i, state in enumerate(path, 1):
        print(f"  {i}. {state}")
    
    # Generate goal stack
    print("\n=== Goal Stack Plan ===\n")
    planner = GoalStackPlanner()
    goals = planner.plan(path)
    
    for goal in goals:
        print(f"{goal['order']}. [{goal['priority']}] {goal['goal']}")
        print(f"   Deadline: {goal['deadline']}")
        for action in goal['actions']:
            print(f"   → {action}")
        print()
