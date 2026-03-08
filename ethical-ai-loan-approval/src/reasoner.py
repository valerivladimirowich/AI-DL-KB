from typing import Dict, List

from fairness_audit import audit_protected_influence


def run_reasoner(concept_scores: Dict[str, float]) -> List[dict]:
    """
    Minimal reasoning layer for the demo pipeline.
    In a full project, replace this with ontology rule execution.
    """
    return audit_protected_influence(concept_scores)
