"""Immutable metric-only feature schema shared by frozen dataset tools."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


FROZEN_PROTOCOL = "dbmags-interaction-benchmark-v10-metric-only-frozen"
FEATURE_SCHEMA_VERSION = "dbmags-metric-time-schema-v1"

# These are the only values a frozen predictor may receive from a raw trace.
FEATURE_METRICS = (
    "mysql_data_locks",
    "mysql_data_lock_waits",
    "mysql_innodb_row_lock_current_waits",
    "mysql_threads_running",
    "mysql_active_sessions",
)
RELATIVE_TIME_BINS_SECONDS = (
    (0.0, 3.0),
    (3.0, 15.0),
    (15.0, 30.0),
    (30.0, 60.0),
    (60.0, 75.0),
)
BASELINE_WINDOW_SECONDS = (-60.0, 0.0)
MINIMUM_BASELINE_SAMPLES = 5
MINIMUM_BIN_SAMPLES = 2

FEATURE_SCHEMA: Dict[str, Any] = {
    "schema_version": FEATURE_SCHEMA_VERSION,
    "metrics": list(FEATURE_METRICS),
    "time_reference": "seconds_relative_to_injection_start",
    "time_bins_seconds": [list(bin_range) for bin_range in RELATIVE_TIME_BINS_SECONDS],
    "baseline_window_seconds": list(BASELINE_WINDOW_SECONDS),
    "baseline_normalization": "(bin_mean - sample_baseline_mean) / max(sample_baseline_std, 1.0)",
    "minimum_baseline_samples": MINIMUM_BASELINE_SAMPLES,
    "minimum_bin_samples": MINIMUM_BIN_SAMPLES,
    "feature_order": "time_bin_major_then_metric",
    "feature_count": len(FEATURE_METRICS) * len(RELATIVE_TIME_BINS_SECONDS),
}


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


FEATURE_SCHEMA_SHA256 = canonical_json_sha256(FEATURE_SCHEMA)
