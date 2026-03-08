from typing import Dict, List
import pandas as pd


PROTECTED_CONCEPTS = {"Age", "Gender", "Race"}


def audit_protected_influence(concept_scores: Dict[str, float], threshold: float = 0.30) -> List[dict]:
    alerts = []
    for concept, score in concept_scores.items():
        if concept in PROTECTED_CONCEPTS and score >= threshold:
            alerts.append(
                {
                    "type": "BiasAlert",
                    "concept": concept,
                    "score": round(score, 4),
                    "message": f"Protected concept '{concept}' has high importance score.",
                }
            )
    return alerts


def approval_rate_by_group(df: pd.DataFrame, group_col: str, prediction_col: str = "predicted_approved") -> Dict[str, float]:
    grouped = df.groupby(group_col)[prediction_col].mean().to_dict()
    return {str(k): float(v) for k, v in grouped.items()}
