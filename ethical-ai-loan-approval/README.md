# Ethical AI for Loan Approval

A professional project skeleton for a loan approval system that combines a deep learning model with a Knowledge Representation (KR) component for fairness and compliance auditing.

## Team Members
- Ara Khachaturov
- Add teammate names here

## Project Goal
This project predicts loan approval outcomes and performs post-prediction fairness auditing using a regulatory ontology. Protected attributes such as **Race**, **Gender**, and **Age** are represented in the ontology, and reasoning rules are used to detect potential bias. If a protected concept receives high feature importance, the system logs a **Bias Alert** in the repository output folder.

## Repository Structure
```text
ethical-ai-loan-approval/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── sample_dataset.csv
│   └── data_description.md
├── src/
│   ├── model.py
│   ├── train.py
│   ├── infer.py
│   ├── explainability.py
│   ├── ontology_mapper.py
│   ├── fairness_audit.py
│   ├── reasoner.py
│   └── utils.py
├── models/
│   └── .gitkeep
├── kb/
│   ├── regulatory_ontology.ttl
│   ├── fairness_rules.txt
│   └── concept_mapping.json
└── output/
    ├── predictions/
    ├── audits/
    └── bias_alerts/
```

## Environment Setup
### Python Version
- Python 3.11 recommended

### Create Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Run
### 1. Train the model
```bash
python src/train.py
```
This trains a simple baseline classifier and saves it into the `models/` folder.

### 2. Run inference + ethical audit
```bash
python src/infer.py
```
This script:
- loads the trained model,
- scores sample applicants,
- computes feature importance,
- maps features to ontology concepts,
- audits fairness constraints,
- writes outputs to:
  - `output/predictions/`
  - `output/audits/`
  - `output/bias_alerts/`

## Integrated KR Component
The KR component is stored in `kb/` and includes:
- a lightweight regulatory ontology,
- fairness rules,
- a feature-to-concept mapping file.

The pipeline is intentionally simple so the repository is easy to understand in an academic setting.

## Brief Results Summary
This repository includes a baseline demonstration pipeline. In a final submission, replace the placeholders with:
- your true model performance,
- fairness metrics,
- examples of generated bias alerts,
- and screenshots or diagrams if needed.

## Suggested Improvements Before Submission
- Replace the sample dataset with your real or approved dataset.
- Add your actual team members.
- Add screenshots of outputs.
- Replace the baseline model with your full deep learning model if required.
- Update the results section with real metrics.

## License
Add a license if required by your course.
