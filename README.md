# Bajaj FraudNet ML

A demonstrable Streamlit prototype for Bajaj ATOM Season 9 PS3: detecting sophisticated, coordinated insurance fraud across interconnected ecosystems at First Notice of Loss (FNOL).

This is the upgraded version. It replaces the earlier rules-heavy prototype with a time-safe hybrid learning system designed to detect subtle fraud while protecting unusual but genuine claims.

## What is implemented

- Supervised behavioural machine learning using `HistGradientBoostingClassifier`
- Chronological train / validation / test split
- Probability calibration on a separate validation period
- Graph-derived historical features for customers, providers, surveyors, TPAs, banks, phones, addresses and devices
- Isolation Forest for novel-pattern detection
- Deterministic evidence controls for known conflicts
- Legitimacy features that reduce false positives
- Green / Amber / Red operating lanes
- Time-safe network visualisation at FNOL
- Local claim explanations with risk-increasing and protective evidence
- Rules-only versus ML comparison
- Precision, recall, PR-AUC, false-positive rate and fraud-value capture
- Explicit unusual-genuine and subtle-fraud stress tests

All records are synthetic and do not represent Bajaj customers.

## Important decision logic

- A high anomaly score alone can route a claim to **Amber** for targeted verification.
- Anomaly alone cannot create a **Red** fraud verdict.
- Red routing requires a high calibrated supervised fraud probability or a decisive deterministic evidence conflict.
- Strong document verification, coherent damage evidence, verified location, catastrophe context and tenure are treated as protective signals.

## Project structure

```text
bajaj-fraudnet-ml/
├── app.py
├── requirements.txt
├── README.md
├── setup_windows.bat
├── run_app.bat
├── verify_install.py
├── smoke_test.py
├── data/
│   └── claims.csv
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── model_engine.py
│   └── graph_engine.py
└── .streamlit/
    └── config.toml
```

## Recommended Python version

Use Python 3.11 or 3.12.

## Fast Windows setup

Open the full project folder in VS Code. In the VS Code terminal, run:

```powershell
.\setup_windows.bat
```

Then launch the app:

```powershell
.\run_app.bat
```

## Manual virtual-environment setup

### 1. Create the environment

```powershell
py -3.11 -m venv .venv
```

Use `py -3.12` instead when Python 3.12 is installed.

### 2. Activate it

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify packages

```powershell
python verify_install.py
```

### 5. Run the smoke test

```powershell
python smoke_test.py
```

### 6. Launch Streamlit

```powershell
python -m streamlit run app.py
```

The app normally opens at `http://localhost:8501`.

## VS Code interpreter

Press `Ctrl + Shift + P`, select **Python: Select Interpreter**, then choose:

```text
.venv\Scripts\python.exe
```

## Recommended live-demo flow

1. Open **FNOL Risk Scanner**.
2. Select **Subtle fraud — normal amount, rotating identities**.
3. Show that the fixed rule score is limited, while the supervised model and graph behaviour identify risk.
4. Select **Unusual but genuine — protected from false positive**.
5. Show the protective legitimacy evidence and Green route.
6. Open **Network Investigator** and display shared ecosystem infrastructure.
7. Open **Model Performance** and compare rules-only performance with the hybrid model.
8. End on **Enterprise Blueprint** to explain the Bajaj-data pilot and production roadmap.

## What the prototype does not claim

- The displayed metrics are not production estimates.
- The model is trained on synthetic proxy data, not Bajaj claim history.
- Exact identifiers stand in for production-grade probabilistic entity resolution.
- Graph neural networks are a future step after sufficient labelled graph history exists.
- The app does not automatically reject real claims.

## Production next steps

- Train on de-identified Bajaj claims and Special Investigation Unit outcomes
- Build probabilistic entity resolution
- Add product- and region-specific models
- Optimise thresholds to investigation capacity and rupee-value capture
- Feed investigator outcomes and appeals back into training
- Monitor feature, prediction and performance drift
- Introduce graph embeddings or GNNs after adequate real labels exist
- Add model registry, role-based access, audit trails and adverse-decision governance
