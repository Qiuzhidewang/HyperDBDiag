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

The 396 multi-root cases are the relevant stress stratum: OpDiag reaches 33.33% Exact, DBAIOps 25.25%, the EPDG hypergraph 56.31%, and complete HyperDBDiag 66.92%. On the two mixed-root occurrence patterns, complete HyperDBDiag reaches 67.80% for sequential injection and 65.15% for overlapping injection. The registered replicate holdout repeats all scenario templates across train and evaluation; it is therefore not an unseen-combination generalization estimate.

The local structured judge retains the 57.58% result while exposing evidence-supported challengers without inventing root sets. With the explicit `gpt-5.5`/`xhigh` Responses configuration, the direct hypergraph-LLM row reaches 57.73% Exact / 74.27% Component F1, and complete HyperDBDiag reaches 65.45% / 79.27%. Training-OOF calibration enabled the direct row in one outer fold and the complete row in all six. The direct path issued 12 test requests and received 12 valid responses; the complete path issued 72 and received 65. Complete arbitration corrected 52 predictions and harmed none. The post-diagnosis advisor used a fixed 24-case, truth-blinded sample: 18 recommendations passed schema and safety checks, five unsafe recommendations were rejected, and diagnosis mutation remained zero. Credentials and raw responses are not retained.
