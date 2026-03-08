from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"


def ensure_directories() -> None:
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "audits").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "bias_alerts").mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
