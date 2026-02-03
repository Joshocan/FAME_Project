# Feature Arguementation Modelling Environment  (FAME) Project

## Setup (Recommended: Virtual Environment)

Using a project-local virtual environment avoids system Python conflicts and ensures consistent dependencies across machines.

### Create the venv

macOS/Linux:
```bash
python3 -m venv .venv
```

Windows (PowerShell):
```powershell
python -m venv .venv
```

### Install requirements

macOS/Linux:
```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r scripts/requirements.txt
```

Windows (PowerShell):
```powershell
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r scripts\requirements.txt
```

### Run scripts and tests (always using the venv Python)

macOS/Linux:
```bash
.venv/bin/python scripts/preprocessing_for_rag.py
.venv/bin/python -m pytest tests/test_ingestion.py -v
```

Windows (PowerShell):
```powershell
.venv\Scripts\python scripts\preprocessing_for_rag.py
.venv\Scripts\python -m pytest tests\test_ingestion.py -v
```

### Optional helper scripts

macOS/Linux:
```bash
./scripts/install_requirements.sh
./scripts/run_ingestion.sh
./scripts/run_tests.sh tests/test_ingestion.py -v
```
