# Experiment protocol and provenance

## Data and evaluation

The root-diagnosis track uses the frozen v11 DB-MAGS-derived cohort in
`data/dbmags_interaction_v11_frozen/`. It contains 660 cases, 25 metric-time features,
seven atomic roots, and 22 scenario templates. There are 264 single-root and
396 two-root cases. Each of six outer folds holds out one replicate index:
550 cases train the model and 110 cases are evaluated once. Scenario templates
remain represented in both partitions.

Feature normalization, training hyperedges, root-set candidates, and evidence
profiles are derived from training data. Case IDs, scenario labels, SQL/object
identities, and ground-truth labels are not supplied to the LLM. The model sees
anonymous evidence packets constructed by `build_packets` in
`src/ecsa_arbitration.py`. SQL and operator identities are retained separately
for the localization task.

The localization track uses `data/dbmags_operator_bound_v4/`: 296 cases and
576 root-specific chains, evaluated with target-template holdout. Its oracle-root
and target-plan conditioning are recorded in the report protocol. These results
should not be multiplied into the root-diagnosis accuracy as if the tracks were
one end-to-end test.

## Parameter selection and calibration

The upstream training procedure selects the raw/propagation weight α, EPDG
weight β, and candidate count K using grouped out-of-fold predictions within
the outer training partition. The metric-selection threshold is θ = 1 in the
reference configuration. The conservative local policy uses five inner
training groups and freshly refitted models to select its profile threshold
and evidence gap. The six selected settings and their training audits are
stored in `runs/hyperdbdiag-unified-20260930/calibration/`; upstream settings
are recorded in `runs/dbmags-ablation/full_report_llm_xhigh.json` and the
sensitivity study's `prepared.json`.

The sensitivity protocol fixes its grid before new sensitivity outcomes. Each
sweep changes only the named parameter while retaining the other fold settings
and calibrated local thresholds. Its 22 configurations include a reference
that must reproduce every main prediction. Results evaluate parameter changes
on the existing development cohort; they are not independent transfer results.

## Saved inference and fallback

The unified experiment regenerates 1,320 primary request packets across the
full-pool and filtered-pool arms. Exact packet hashes must match the recorded
inputs in the full-coverage ECSA experiment. Replay preserves the original
primary decision or failure; repeat draws do not replace unsuccessful primary
outcomes. The complete method falls back to its calibrated local prediction,
and the full-pool LLM ablation falls back to the upstream top-1 prediction.

The evidence stress study retains visible structured model responses, including
invalid responses, and rechecks them with the validator. Its clean repeat is a
separate draw. Zero final out-of-pool predictions describes the output of
validation and fallback over the tested cases.

The registered model string is `gpt-5.5`, with reasoning effort `xhigh`; recorded
transport metadata are included in the corresponding run folders. Temperature,
a decoding seed, and an immutable service-side model snapshot were not recorded.
The seed `20260802` governs deterministic experiment operations and should not
be presented as a decoding seed.

## Reproduction boundaries

`reproduce.py --verify-existing` checks saved files and recomputes reported
metrics without modifying run records. The default refit rebuilds the current
upstream model and local calibration and checks all primary request packets
before replay. OpDiag, DBAIOps, and unchanged lower-level ablation predictions
are carried forward from their recorded reports. Separate localization and
structure entrypoints are listed in the root README.

Historical run directories and `archives/pre-unification-20260930.zip` preserve
original protocols and source hashes. Some historical audits also record hashes
of manuscript files that are outside this repository. Their original values
are retained; current release verification checks the distributed experimental
inputs without requiring those private manuscript snapshots.

The only change to the unified experiment source in this repository update is
archive-path resolution for relocated checkouts. Its original bytes and the
old/new hashes are preserved in `docs/source_provenance.json`. Model fitting,
calibration, request construction, and metric definitions are unchanged.
