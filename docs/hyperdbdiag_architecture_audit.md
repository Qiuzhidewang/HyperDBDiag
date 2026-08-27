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

The clean rerun reproduced every one of the old 660 deterministic predictions. On the shared KPI-only track, OpDiag, DBAIOps, and the hypergraph without EPDG reach 84.85%, 85.00%, and 85.61% Exact, respectively. Hypergraph Exact increases from 85.61% without EPDG to 90.61% with EPDG; component F1 increases from 91.00% to 94.13%. This latter gain includes the available anonymous SQL-shape evidence and is not a pure structure-only comparison. EPDG corrects 36 cases and harms three, with positive net gain in all six folds.

The 396 multi-root cases are the relevant stress stratum: OpDiag 81.57% Exact, DBAIOps 81.06%, KPI-only hypergraph 85.61%, and EPDG + hypergraph 90.66%. The registered replicate holdout repeats all scenario templates across train and evaluation; it is therefore not an unseen-combination generalization estimate.

The local structured judge retains the 90.61% result. It can expose an evidence-supported challenger but cannot invent a root set. With the explicit `gpt-5.5`/`xhigh` Responses configuration, the direct hypergraph-LLM row reaches 90.76% Exact / 94.22% Component F1, and the complete HyperDBDiag row reaches 91.97% / 94.98%. Training-OOF calibration enabled the direct row in one outer fold and the complete row in five. The direct path issued 12 test requests and received 12 valid responses; the complete path issued 55 and received 39, with transport failures falling back without mutation. The post-diagnosis advisor used a fixed 24-case, truth-blinded sample: 17 recommendations passed schema and safety checks, three unsafe recommendations were rejected, and diagnosis mutation remained zero. Credentials and raw responses are not retained.
