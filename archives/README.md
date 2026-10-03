# Historical source archive

`pre-unification-20260930.zip` contains exact source, tests and README bytes
from before the unified ECSA experiment and cleanup. Its SHA-256 is recorded in
`runs/hyperdbdiag-unified-20260930/before_manifest.json` and verified by the new
experiment. Historical run data, model decisions, protocols and reports remain
in their original directories.

The seven retired pilot experiment/analysis scripts and two old test files can
be recovered from this archive. Their shared request builder and validator now
live in `src/ecsa_arbitration.py`; the current run is `src/ecsa_experiment.py`.

For historical reproduction, extract into a separate workspace with the
corresponding frozen data and run records. Do not overwrite the current source
checkout. Some historical protocols also freeze manuscript bytes. Those manuscript snapshots are not distributed in this repository. Historical
fingerprints have not been rewritten to match later manuscript edits. Use the
current read-only verifier for the distributed experimental inputs.

`source-before-github-sync/src/ecsa_experiment.py` preserves the original unified
runner before its archive-path portability fix. The original and current hashes
are recorded in `docs/source_provenance.json`; experiment logic is unchanged.
