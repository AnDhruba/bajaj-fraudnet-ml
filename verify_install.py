from __future__ import annotations

import importlib
import sys

PACKAGES = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "networkx": "networkx",
    "scikit-learn": "sklearn",
    "plotly": "plotly",
}

failed = []
for label, module in PACKAGES.items():
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "installed")
        print(f"[OK] {label}: {version}")
    except Exception as exc:
        failed.append((label, str(exc)))
        print(f"[FAIL] {label}: {exc}")

if failed:
    print("\nSome required packages are missing or broken.")
    sys.exit(1)

print("\nAll required packages are installed.")
