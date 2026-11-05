# HyperDBDiag

HyperDBDiag identifies multiple root causes of database performance anomalies
and localizes roots with execution evidence to SQL statements and operators.
It combines an EPDG-enhanced metric hypergraph with evidence-constrained set
arbitration (ECSA): a calibrated local discriminator filters candidate root
sets, and an LLM arbitrates within the retained candidates.

This public repository contains the implementation, frozen data, experiment
records, and reproduction tools for the manuscript **HyperDBDiag: Multi-Root-Cause
Identification and Operator-Level Localization for Database Performance
Anomalies**, prepared for the **ICDE 2027 Research Track**.

The current root-diagnosis results are in
[`runs/hyperdbdiag-unified-20260930/`](runs/hyperdbdiag-unified-20260930/).
These records support the results reported in the ICDE manuscript. Earlier
runs are retained for provenance and are identified below.

## Quick start

Download the repository using **Code → Download ZIP** on GitHub and extract it,
or clone it without signing in:

```bash
git clone https://github.com/Qiuzhidewang/HyperDBDiag.git
cd HyperDBDiag
```

Python **3.9.25** was used for validation. Run the following commands from the
downloaded repository root:

```bash
python3.9 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# Run the tests and verify the recorded results without refitting.
.venv/bin/python reproduce.py --verify-existing

# Refit the current root-diagnosis pipeline and replay frozen LLM responses.
.venv/bin/python reproduce.py --output reproduced/current
```

Both commands run offline after dependency installation and require no API
key, model login, or running database. Verification recomputes metrics from
case-level predictions, checks recorded hashes, and validates the saved
responses in the evidence-perturbation experiment. It leaves the recorded
results unchanged.

The refit command rebuilds the upstream models and training-side local
calibration for six outer folds, verifies the regenerated candidate sets and
request packets, and replays the exact saved primary LLM responses, including
recorded failures. Unchanged baseline predictions are loaded from their frozen
reports. Sensitivity, evidence-perturbation, localization, and structure reports
are checked from saved records. This command does not make fresh LLM calls;
its execution time is not an end-to-end inference latency measurement.

New results go to a fresh directory under `reproduced/`. Choose another output
directory for subsequent runs; the runner rejects a nonempty directory and
protects `runs/`, `data/`, and source files. To verify a generated run:

```bash
.venv/bin/python reproduce.py --verify-existing --output reproduced/current
```

Use `--skip-tests` after a successful test run. The standalone test command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m unittest discover -s tests -v
```

## Current results

All values below are percentages. Exact measures unordered root-set equality;
Precision, Recall, and F1 aggregate root components across cases.

| Method | Exact | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| OpDiag reproduction | 42.58 | 66.02 | 64.77 | 65.39 |
| DBAIOps reproduction | 33.33 | 60.94 | 60.42 | 60.68 |
| HyperDBDiag | **87.88** | **94.81** | **93.37** | **94.08** |

The comparison uses 660 cases and six fixed replicate-index holdouts, with
550 training and 110 evaluation cases per fold. HyperDBDiag obtains 91.67%
Exact on single-root cases and 85.35% on two-root cases; sequential and
overlapping two-root cases reach 85.98% and 84.09%, respectively.

### Component ablation

| Variant | Exact | F1 |
| --- | ---: | ---: |
| Ordinary pairwise graph | 28.48 | 64.01 |
| Metric hypergraph without EPDG | 40.61 | 63.08 |
| EPDG-enhanced metric hypergraph | 57.58 | 74.17 |
| EPDG + local discriminator, without LLM | 82.73 | 90.12 |
| EPDG + LLM, without local filtering | 87.58 | 93.86 |
| HyperDBDiag | **87.88** | **94.08** |

The local-only row uses the calibrated local prediction. The LLM-only row
arbitrates over the upstream candidate pool, while the complete method uses
the locally retained pool. Rejected or unavailable LLM decisions fall back to
the upstream prediction in the LLM-only row and to the local prediction in
the complete method. A separate comparison with a shared upstream fallback
is retained in the case records to examine the filtering effect.

### SQL/operator localization and structure

| Measure | OpDiag reproduction | HyperDBDiag |
| --- | ---: | ---: |
| SQL Hit@1 | 50.87 | 93.75 |
| SQL Hit@2 | 80.21 | 99.13 |
| Operator Hit@1 | 75.00 | 91.67 |
| Joint SQL-Top-2/operator-Top-1 | 57.29 | 90.80 |

Localization is evaluated separately on 296 cases with 576 root-to-SQL/operator
chains. The ranker is supplied with the ground-truth root, and operator ranking
is evaluated within the target SQL plan. These are conditional localization
results, not end-to-end diagnosis-and-localization accuracy.

The hypergraph stores 79.90% fewer relation items than the equivalent pairwise
representation. The recorded traversal-time reduction is 79.77%; these timings
cover the graph traversal operation and depend on the machine.

### Sensitivity and evidence perturbations

The sensitivity study varies one parameter at a time:

| Parameter | Evaluated values |
| --- | --- |
| Raw/propagation weight α | 0, 0.2, 0.4, 0.6, 0.8, 1 |
| EPDG weight β | 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1 |
| Metric-selection threshold θ | 0.5, 0.75, 1, 1.25, 1.5 |
| Candidate count K | 1, 2, 3, 4, 6, 8 |

There are 22 distinct configurations including the reference, each evaluated
on all 660 cases. The reference predictions exactly match the main experiment.
See the [sensitivity figure](runs/parameter-sensitivity-20260930-v1/parameter_sensitivity.pdf)
and [case-level results](runs/parameter-sensitivity-20260930-v1/predictions.json).

The evidence experiment evaluates seven conditions on all 660 cases: clean
cards, identical support across candidates, empty counterevidence, shuffled
support/counterevidence, changed order, an out-of-pool lure, and a separate
clean repeat. All 4,620 final predictions remain within the supplied candidate
pool after validation and fallback. The clean condition is a separate draw
(88.03% Exact) and does not replace the primary result above. Details are in
the [evidence report](runs/ecsa-evidence-stress-full660-20260930-v1/results.json).

## Paper-to-artifact map

Table and figure numbers below follow the ICDE manuscript. The linked reports
contain the recorded numerical results; plot files in historical run folders
retain their original rendering.

| Paper result | Recorded source |
| --- | --- |
| Table I: root-cause diagnostic performance | [`main_comparison.json`](runs/hyperdbdiag-unified-20260930/main_comparison.json) |
| Table I (SQL/operator columns) and Figure 6: SQL/operator localization | [`full_report.json`](runs/dbmags-operator-bound-v4/full_report.json) |
| Table II: sequential and overlapping root causes | `by_occurrence_pattern` in [`main_comparison.json`](runs/hyperdbdiag-unified-20260930/main_comparison.json) |
| Figure 5: single-root and mixed-root diagnosis | `by_root_cardinality` in [`main_comparison.json`](runs/hyperdbdiag-unified-20260930/main_comparison.json); [recorded plot](runs/hyperdbdiag-unified-20260930/figure5_updated.pdf) |
| Table III: graph size and traversal | [`full_report.json`](runs/dbmags-structure-efficiency/full_report.json) |
| Table IV: six component variants | [`ablation.json`](runs/hyperdbdiag-unified-20260930/ablation.json) |
| Figure 7: parameter sensitivity | [`results.json`](runs/parameter-sensitivity-20260930-v1/results.json) |
| Table V: evidence perturbations | [`results.json`](runs/ecsa-evidence-stress-full660-20260930-v1/results.json) |

## Protocol and LLM records

The root-diagnosis data contain 25 metric-time features, seven atomic roots,
and 22 scenario templates. Splits retain known scenarios in training and
evaluation. Candidate root sets come from combinations in the training fold;
this evaluation does not measure transfer to unseen root combinations or a
new workload. Upstream parameter selection and local calibration use the
outer training data. See the [protocol notes](docs/experiment_protocol.md)
for selection, data boundaries, and run provenance.

The recorded model setting is `gpt-5.5` with `xhigh` reasoning effort. The exact
instructions, packet builder, output schema, and validator are in
[`src/ecsa_arbitration.py`](src/ecsa_arbitration.py). Run protocols retain their
hashes and transport metadata. Temperature and a decoding seed were not
explicitly set; the experiment seed controls data/model operations, not LLM
sampling. The reported LLM results are saved point estimates. Offline replay
preserves these estimates without claiming that a fresh service call will
return identical text or predictions.

Historical live-run scripts are included for provenance. They require a
separately configured authenticated model client; the saved CLI transport
contains a workstation-specific executable path. The offline commands above
are the supported portable entrypoints and never invoke that transport.

## Additional offline entrypoints

These commands refit the separate localization experiment and rerun the
structure comparison into new files:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  src/operator_bound_experiment.py \
  --dataset-root data/dbmags_operator_bound_v4 --audit-only

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  src/operator_bound_experiment.py \
  --dataset-root data/dbmags_operator_bound_v4 \
  --output reproduced/operator.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  src/structure_efficiency_experiment.py --output reproduced/structure.json
```

Plotting scripts additionally use `requirements-plots.txt`. Historical analysis
scripts can write into their supplied output directory, and some audits bind
old manuscript snapshots. Use `reproduce.py --verify-existing` to inspect the
current release without changing its frozen reports.

## Repository layout and history

- `src/`: models, evidence handling, and experiment entrypoints.
- `tests/`: implementation and data-contract tests.
- `data/`: frozen diagnostic features, evidence, and localization data.
- `runs/`: protocols, case-level predictions, model responses, and reports.
- `scripts/`: supplementary experiments and read-only release verification.
- `docs/`: experiment protocol, architecture, data-source audit, and source provenance.
- `archives/`: exact historical source retained for hash verification.
- `reproduce.py`: offline verification and refit runner.
- `MANIFEST.sha256`: checksums for the distributed files.

The older `runs/dbmags-ablation/` and `runs/dbmags-main-comparison/` results describe the
previous arbitration policy, including the historical **65.45% Exact** result.
They supply unchanged baseline records and preserve the development history;
they are not the current complete-method result. The `ecsa-isolation`,
`ecsa-conservative`, and `ecsa-full660` directories record subsequent component
experiments. The current unified run is the reference for the tables above.

A portability fix resolves historical absolute source paths against archive
members without altering the recorded protocols or hashes. The original source
and both file hashes are documented in
[`docs/source_provenance.json`](docs/source_provenance.json).

DB-MAGS fault definitions were audited against the
[upstream project](https://github.com/qifeng1128/DB-MAGS); its checkout and a
running database are not required for the frozen-data experiments. Code is
provided under the [MIT License](LICENSE).

This is the public artifact link for the ICDE manuscript. Access to the
distributed files does not require a GitHub account. The supported offline
workflow does not require an API key or model-service access after dependency
installation. Historical files and commit messages retain their original
dates and terminology; experiment records and their recorded hashes are
preserved for traceability.
