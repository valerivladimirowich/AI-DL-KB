import json
from pathlib import Path
from typing import Dict

from utils import BASE_DIR


MAPPING_FILE = BASE_DIR / "kb" / "concept_mapping.json"


def load_mapping() -> dict:
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def map_features_to_ontology(feature_importance: Dict[str, float]) -> Dict[str, float]:
    mapping = load_mapping()
    concept_scores: Dict[str, float] = {}

    for feature_name, score in feature_importance.items():
        concept = mapping.get(feature_name)
        if concept:
            concept_scores[concept] = concept_scores.get(concept, 0.0) + score

    return concept_scores
