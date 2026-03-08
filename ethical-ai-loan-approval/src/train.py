from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model import build_model
from utils import DATA_DIR, MODEL_DIR, ensure_directories


def main() -> None:
    ensure_directories()
    df = pd.read_csv(DATA_DIR / "sample_dataset.csv")

    X = df.drop(columns=["approved"])
    y = df["approved"]

    numeric_features = ["income", "credit_score", "loan_amount", "employment_years", "age"]
    categorical_features = ["gender", "race"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_model()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    output_path = MODEL_DIR / "loan_model.joblib"
    joblib.dump(pipeline, output_path)

    print(f"Model saved to: {output_path}")
    print(f"Baseline accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
