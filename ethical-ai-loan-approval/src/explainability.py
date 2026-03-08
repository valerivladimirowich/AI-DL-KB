from typing import Dict
import numpy as np


def get_feature_importance(pipeline) -> Dict[str, float]:
    """
    Lightweight importance extraction for the baseline logistic regression model.
    Returns absolute coefficient magnitudes mapped to transformed feature names.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = np.abs(model.coef_[0])
    return {name: float(weight) for name, weight in zip(feature_names, coefficients)}
