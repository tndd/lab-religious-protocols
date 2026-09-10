# lab-religious-protocols

Executable research environment for the Religious Protocols project.

## Purpose

This repository contains reproducible code, tests, synthetic benchmarks, and experiment entry points. The conceptual research record, literature notes, and paper history live separately in the Lab workspace.

## First Kaggle smoke test

```bash
python -m pip install -e .
pytest -q
python experiments/synthetic_001.py
```

Results are written to `results/`.
