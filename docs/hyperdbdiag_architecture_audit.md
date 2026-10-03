# HyperDBDiag Active Architecture

The artifact now has one paper-facing dataset path: frozen DB-MAGS input under a six-fold leave-one-replicate-index-out evaluation. The 22 scenarios remain represented in every outer train and evaluation partition. Scenario and table provenance are audit-only and are never predictors.

## Model Boundary

The ordinary graph and hypergraph receive identical outer-training KPI rows and labels but share no fitted model state. The ordinary baseline builds a training-fold Pearson kNN metric graph and independent root classifiers. The hypergraph splits standardized metrics into positive and negative atoms, retains each outer-training sample as one hyperedge, and applies static H propagation plus query-dependent aggregation over the frozen training hyperedges. A held-out query cannot change H.

The main comparison also includes a separate DBAIOps reproduction. Its training-fold anomaly models derive multi-metric statistics, its ExperienceGraph contains trigger/metric/experience/tag vertices, and its graph evolution performs metric-proximity expansion followed by abnormal-state clipping. It uses no HyperDBDiag candidate inventory or fitted state; the closed DB-MAGS root inventory is decoded with a training-fold Top-k root-count model.

EPDG root-metric paths weight training incidence. Stable signed paths and all fusion weights are learned or selected inside the current outer training fold by grouped OOF evaluation. Anonymous SQL-shape atoms are frozen and hash-bound to opaque case IDs; only registered direct root-to-atom paths can contribute a query-time prior. They are not SQL identities, plans, or operator targets. Raw SQL, identifiers, counts, full plans, source IDs, scenario fields, and evaluation labels are excluded.

## Evaluation Protocol

| Property | Registered value |
| --- | --- |
| Cases | 660 |
| Features | 25 metric-time values |
| Roots | 7 DB-MAGS atomic roots |
| Outer folds | 6 replicate-index holdouts |
| Train/evaluation per fold | 550 / 110 |
| Single/double-root cases | 264 / 396 |

The registered leave-one-replicate-index-out path is the only active outer protocol. It retains all 22 scenario templates in both partitions and therefore measures repeated-block diagnosis, not unseen-scenario transfer. Unseen-scenario results are not mixed into the main comparison.

## Verified Result

The audited v11 rerun uses clean recovery between all 660 collections. On the shared KPI-only track, OpDiag, DBAIOps, and the hypergraph without EPDG reach 42.58%, 33.33%, and 40.61% Exact, respectively. Hypergraph Exact increases from 40.61% without EPDG to 57.58% with EPDG; component F1 increases from 63.08% to 74.17%. This latter gain includes registered anonymous SQL-shape and runtime lock-wait evidence and is not a pure structure-only comparison. EPDG corrects 127 cases and harms 15, with positive net gain in all six folds.

The current unified experiment uses the calibrated conservative local policy. Its
local-only variant reaches 82.73% Exact / 90.12% Component F1; LLM arbitration
without local filtering reaches 87.58% / 93.86%; and complete HyperDBDiag reaches
87.88% / 94.08%. The complete method reaches 85.35% Exact on the 396 two-root
cases, with 85.98% for sequential and 84.09% for overlapping occurrences.

Local calibration uses training-only inner groups. The LLM reviews each held-out
case using the registered `gpt-5.5`/`xhigh` configuration. The unified runner
refits the deterministic pipeline and replays exact saved primary inputs and
responses; invalid or unavailable responses retain the calibrated local
prediction. The separate LLM-only variant uses the upstream top-1 fallback.

The older 65.45% Exact result and limited-call policy remain in the historical
reports. Current main and ablation results are in
`runs/hyperdbdiag-unified-20260930/`. Supplementary sensitivity and full-cohort
evidence perturbations are linked from the root README. See
[experiment_protocol.md](experiment_protocol.md) for reproduction boundaries.
