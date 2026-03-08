from datetime import datetime, UTC
import joblib
import pandas as pd

from explainability import get_feature_importance
from ontology_mapper import map_features_to_ontology
from reasoner import run_reasoner
from fairness_audit import approval_rate_by_group
from utils import DATA_DIR, MODEL_DIR, OUTPUT_DIR, ensure_directories, write_json


def main() -> None:
    ensure_directories()

    model_path = MODEL_DIR / "loan_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run 'python src/train.py' first.")

    pipeline = joblib.load(model_path)
    df = pd.read_csv(DATA_DIR / "sample_dataset.csv")

    X = df.drop(columns=["approved"])
    df["predicted_approved"] = pipeline.predict(X)
    df["prediction_probability"] = pipeline.predict_proba(X)[:, 1]

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    predictions_path = OUTPUT_DIR / "predictions" / f"predictions_{timestamp}.csv"
    df.to_csv(predictions_path, index=False)

    feature_importance = get_feature_importance(pipeline)
    concept_scores = map_features_to_ontology(feature_importance)
    alerts = run_reasoner(concept_scores)

    audit_payload = {
        "timestamp": timestamp,
        "feature_importance": feature_importance,
        "concept_scores": concept_scores,
        "approval_rate_by_gender": approval_rate_by_group(df, "gender"),
        "approval_rate_by_race": approval_rate_by_group(df, "race"),
    }
    write_json(OUTPUT_DIR / "audits" / f"audit_{timestamp}.json", audit_payload)

    if alerts:
        write_json(
            OUTPUT_DIR / "bias_alerts" / f"bias_alert_{timestamp}.json",
            {"timestamp": timestamp, "alerts": alerts},
        )
        print("Bias alert generated.")
    else:
        print("No bias alert generated.")

    print(f"Predictions saved to: {predictions_path}")
    print(f"Audit summary saved to: {OUTPUT_DIR / 'audits' / f'audit_{timestamp}.json'}")


if __name__ == "__main__":
    main()
