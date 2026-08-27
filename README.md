# HyperDBDiag Reproducibility Artifact

This repository is the reproducibility artifact prepared for the HyperDBDiag
VLDB 2027 submission. It contains the implementation, frozen DB-MAGS-derived
data, registered reports, and tests used for the experiments in the paper.

## Environment

The artifact was tested with Python 3.9.25 in the `rca39` environment. A
clean environment can be created as follows:

```bash
python3.9 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The default reproduction is offline and does not require an API key. The
optional `requirements-llm.txt` file is only needed for a new live model run;
the registered results are already included.

## Reproduce the Results

Run from this directory:

```bash
.venv/bin/python reproduce.py
```

This command runs the tests, root-cause comparison, ablation, SQL/operator
evaluation, and graph-structure comparison. New files are written to
`reproduced/`; the registered reports in `runs/` are left unchanged. Use
`--skip-tests` when repeating the experiments after the first successful run.

The main registered results are:

| Experiment | Result |
| --- | --- |
| Root-cause Exact: OpDiag / DBAIOps / HyperDBDiag | 84.85% / 85.00% / 91.97% |
| SQL Hit@1: OpDiag / HyperDBDiag | 50.87% / 93.75% |
| Operator Hit@1: OpDiag / HyperDBDiag | 75.00% / 91.67% |
| Hypergraph relation reduction | 62.04% |

Traversal time is machine-dependent; the registered comparison reports a
56.82% reduction.

## Individual Entrypoints

Set `PYTHONPATH=src` when running an entrypoint directly:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  src/hyperdbdiag_ablation.py --seed 20260802 \
  --output reproduced/ablation.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  src/dbmags_experiment.py --seed 20260802 \
  --canonical-method epdg_hypergraph \
  --ablation-report reproduced/ablation.json \
  --output reproduced/main_deterministic.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  src/operator_bound_experiment.py \
  --dataset-root data/dbmags_operator_bound_v4 --audit-only

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  src/operator_bound_experiment.py \
  --dataset-root data/dbmags_operator_bound_v4 \
  --output reproduced/operator.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  src/structure_efficiency_experiment.py \
  --output reproduced/structure.json
```

## Contents

- `src/`: implementation and experiment entrypoints.
- `tests/`: tests for the implementation and data contracts.
- `data/`: frozen DB-MAGS-derived evaluation data.
- `runs/`: registered experiment reports.
- `docs/`: data-source and architecture notes.
- `reproduce.py`: full reproduction runner.
- `MANIFEST.sha256`: file checksums.

The official DB-MAGS source used for mechanism auditing is available at
<https://github.com/qifeng1128/DB-MAGS>. It is not required for reproducing
the frozen results.
