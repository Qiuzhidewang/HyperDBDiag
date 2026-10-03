"""Freeze collected DB-MAGS cases into label-isolated experiment inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from metric_frozen_schema import (
    BASELINE_WINDOW_SECONDS,
    FEATURE_METRICS,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_SHA256,
    FROZEN_PROTOCOL,
    MINIMUM_BASELINE_SAMPLES,
    MINIMUM_BIN_SAMPLES,
    RELATIVE_TIME_BINS_SECONDS,
)
from sql_semantic_evidence import FROZEN_SEMANTIC_PROTOCOL


DEFAULT_SOURCE = Path("data/dbmags_interaction_v11_recollected")
DEFAULT_OUTPUT = Path("data/dbmags_interaction_v11_frozen")
DEFAULT_SEED = 20260802
SOURCE_PROTOCOL = "dbmags-interaction-v11-window-complete"
CONDITIONS = frozenset({"single_a", "single_b", "a_then_b", "b_then_a", "overlap"})
CASE_NAME = re.compile(
    r"^(?P<scenario>.+)__block_(?P<replicate>[0-9]+)__(?P<condition>single_a|single_b|a_then_b|b_then_a|overlap)$"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=True,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _mean(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"metric {metric} has no finite samples")
    return float(np.mean(values))


def metric_time_features(payload: Mapping[str, Any]) -> Tuple[float, ...]:
    baseline_start, baseline_end = BASELINE_WINDOW_SECONDS
    baseline = [
        row
        for row in payload.get("baseline") or ()
        if baseline_start <= float(row["relative_seconds"]) < baseline_end
    ]
    anomaly = list(payload.get("anomaly") or ())
    if len(baseline) < MINIMUM_BASELINE_SAMPLES:
        raise ValueError("insufficient baseline samples")

    baseline_mean = {metric: _mean(baseline, metric) for metric in FEATURE_METRICS}
    baseline_scale = {
        metric: max(
            float(np.std([float(row[metric]) for row in baseline], dtype=np.float64)),
            1.0,
        )
        for metric in FEATURE_METRICS
    }
    features = []
    for start, end in RELATIVE_TIME_BINS_SECONDS:
        rows = [row for row in anomaly if start <= float(row["relative_seconds"]) < end]
        if len(rows) < MINIMUM_BIN_SAMPLES:
            raise ValueError(f"insufficient samples in anomaly bin {start:g}-{end:g}s")
        features.extend(
            (_mean(rows, metric) - baseline_mean[metric]) / baseline_scale[metric]
            for metric in FEATURE_METRICS
        )
    result = tuple(float(value) for value in features)
    if len(result) != int(FEATURE_SCHEMA["feature_count"]) or not np.all(np.isfinite(result)):
        raise ValueError("invalid frozen metric feature vector")
    return result


def _anonymous_observations(payload: Mapping[str, Any]) -> list[Dict[str, list[str]]]:
    observations: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Dict[str, list[str]]] = {}
    for row in payload.get("observations") or ():
        atoms = tuple(sorted({str(value) for value in row.get("atoms") or () if str(value)}))
        channels = tuple(
            sorted({str(value) for value in row.get("source_channels") or () if str(value)})
        )
        if not atoms or not channels:
            raise ValueError("semantic observations must contain atoms and source channels")
        observations[(atoms, channels)] = {
            "atoms": list(atoms),
            "source_channels": list(channels),
        }
    return [observations[key] for key in sorted(observations)]


def _expected_from_schedule(case: Mapping[str, Any]) -> Tuple[str, ...]:
    collection = case.get("collection") or {}
    condition = str(collection.get("condition") or "")
    schedule = collection.get("schedule") or ()
    roots = tuple(sorted({str(row.get("root") or "") for row in schedule}))
    if condition not in CONDITIONS or not roots or "" in roots:
        raise ValueError("invalid collection schedule")
    expected_count = 1 if condition in {"single_a", "single_b"} else 2
    if len(roots) != expected_count:
        raise ValueError("schedule and condition root counts differ")
    return roots


def _source_inventory(source: Path) -> list[Path]:
    case_root = source / "cases"
    paths = sorted(path for path in case_root.iterdir() if path.is_dir())
    progress = _read_json(source / "progress.json")
    if (
        progress.get("protocol") != SOURCE_PROTOCOL
        or int(progress.get("requested_case_count", 0)) != len(paths)
        or int(progress.get("completed_case_count", 0)) != len(paths)
        or int(progress.get("failed_case_count", -1)) != 0
    ):
        raise ValueError("source collection is incomplete")
    return paths


def freeze(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    source, output = Path(source), Path(output)
    protocol = _read_json(source / "protocol.json")
    if protocol.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("unexpected source collection protocol")
    paths = _source_inventory(source)

    records = []
    source_hashes = []
    condition_counts: Counter[str] = Counter()
    complete_evidence_counts: Counter[str] = Counter()
    mixed_case_count = 0
    complete_mixed_count = 0
    scenario_conditions: Counter[Tuple[str, int, str]] = Counter()
    for path in paths:
        match = CASE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"malformed source case name: {path.name}")
        case_path = path / "case.json"
        metric_path = path / "metrics.json"
        semantic_path = path / "semantic_observations.json"
        plan_path = path / "plans.json"
        trace_path = path / "statement_trace.json"
        if any(not item.is_file() for item in (case_path, metric_path, semantic_path, plan_path, trace_path)):
            raise ValueError(f"source case is missing an artifact: {path.name}")
        case = _read_json(case_path)
        if case.get("protocol") != SOURCE_PROTOCOL or case.get("status") != "collected":
            raise ValueError(f"source case did not pass collection: {path.name}")
        collection = case.get("collection") or {}
        condition = match.group("condition")
        replicate = int(match.group("replicate"))
        scenario = match.group("scenario")
        if (
            str(case.get("case_id")) != path.name
            or str(collection.get("condition")) != condition
            or int(collection.get("replicate_index", 0)) != replicate
        ):
            raise ValueError(f"source case metadata differs from its path: {path.name}")
        expected = _expected_from_schedule(case)
        truth = tuple(sorted(str(value) for value in (case.get("truth") or {}).get("expected_labels") or ()))
        if truth != expected:
            raise ValueError(f"source truth differs from injection schedule: {path.name}")
        if not (case.get("workload_audit") or {}).get("passed") or not (
            case.get("recovery_audit") or {}
        ).get("passed"):
            raise ValueError(f"source runtime audit failed: {path.name}")

        semantic_payload = _read_json(semantic_path)
        observations = _anonymous_observations(semantic_payload)
        observed_atoms = sorted({atom for row in observations for atom in row["atoms"]})
        evidence_audit = case.get("evidence_audit") or {}
        if observed_atoms != sorted(str(value) for value in evidence_audit.get("observed_atoms") or ()):
            raise ValueError(f"semantic evidence audit differs from frozen atoms: {path.name}")
        evidence_by_root = evidence_audit.get("by_root") or {}
        complete = all(evidence_by_root.get(root) is True for root in expected)
        if not complete:
            raise ValueError(f"a scheduled root lacks structural evidence: {path.name}")

        condition_counts[condition] += 1
        complete_evidence_counts[condition] += int(complete)
        scenario_conditions[(scenario, replicate, condition)] += 1
        if len(expected) == 2:
            mixed_case_count += 1
            complete_mixed_count += int(complete)
        records.append(
            {
                "source_case_id": path.name,
                "source_block_id": f"{scenario}__block_{replicate:02d}",
                "scenario": scenario,
                "table": str(collection.get("table") or ""),
                "replicate": replicate,
                "condition": condition,
                "expected": expected,
                "features": metric_time_features(_read_json(metric_path)),
                "observations": observations,
            }
        )
        source_hashes.append(
            {
                "source_case_id": path.name,
                "case_sha256": _sha256(case_path),
                "metrics_sha256": _sha256(metric_path),
                "semantic_observations_sha256": _sha256(semantic_path),
                "plans_sha256": _sha256(plan_path),
                "statement_trace_sha256": _sha256(trace_path),
            }
        )

    scenarios = {row["scenario"] for row in records}
    replicates = {int(row["replicate"]) for row in records}
    if (
        len(records) != 660
        or len(scenarios) != 22
        or replicates != set(range(1, 7))
        or set(condition_counts) != CONDITIONS
        or set(condition_counts.values()) != {132}
        or set(scenario_conditions.values()) != {1}
        or len(scenario_conditions) != len(scenarios) * len(replicates) * len(CONDITIONS)
    ):
        raise ValueError("source cohort does not match the registered 22-by-6-by-5 design")

    random.Random(int(seed)).shuffle(records)
    inputs, folds, truth_rows, provenance, semantic_rows = [], [], [], [], []
    for index, row in enumerate(records, start=1):
        case_id = f"case-{index:04d}"
        inputs.append({"case_id": case_id, "metric_time_features": list(row["features"])})
        folds.append({"case_id": case_id, "block_index": row["replicate"]})
        truth_rows.append({"case_id": case_id, "expected_labels": list(row["expected"])})
        provenance.append(
            {
                "case_id": case_id,
                "source_case_id": row["source_case_id"],
                "source_block_id": row["source_block_id"],
                "source_replicate_index": row["replicate"],
                "source_scenario": row["scenario"],
                "source_table": row["table"],
            }
        )
        semantic_rows.append({"case_id": case_id, "observations": row["observations"]})

    labels = sorted({label for row in records for label in row["expected"]})
    source_audit = {
        "source_protocol": SOURCE_PROTOCOL,
        "raw_root": str(source),
        "protocol_sha256": _sha256(source / "protocol.json"),
        "progress_sha256": _sha256(source / "progress.json"),
        "case_count": len(records),
        "protocol_complete": True,
        "mechanism_stress_ready": complete_mixed_count == mixed_case_count,
        "evidence_completeness_summary": {
            "physical_block_count": len(scenarios) * len(replicates),
            "mixed_case_count": mixed_case_count,
            "mixed_cases_with_complete_root_evidence": complete_mixed_count,
            "mixed_cases_missing_root_evidence": mixed_case_count - complete_mixed_count,
            "condition_case_counts": dict(sorted(condition_counts.items())),
            "condition_complete_evidence_counts": dict(sorted(complete_evidence_counts.items())),
        },
        "cohort_policy": "all planned cases retained; no post-collection filtering",
        "feature_source": "only cases/<source_case_id>/metrics.json",
        "semantic_source": "only anonymous atoms and channels from semantic_observations.json",
        "truth_source": "only the registered injection schedule, cross-checked against case truth",
        "fold_source": "only the registered replicate index",
        "predictor_excluded": [
            "source_case_id",
            "scenario",
            "condition",
            "root labels",
            "raw SQL",
            "execution plans",
            "statement signatures",
            "anomaly scores",
            "injection and audit records",
        ],
        "source_file_sha256": source_hashes,
    }
    payloads = {
        "frozen_inputs.json": {
            "protocol": FROZEN_PROTOCOL,
            "feature_schema": FEATURE_SCHEMA,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "samples": inputs,
        },
        "fold_manifest.json": {
            "protocol": FROZEN_PROTOCOL,
            "inner_oof_group": {
                "name": "replicate_index_cross_fit",
                "group_field": "block_index",
                "purpose": "training_fold_h_residual_construction_only",
            },
            "samples": folds,
        },
        "ground_truth.json": {
            "protocol": FROZEN_PROTOCOL,
            "label_inventory": labels,
            "samples": truth_rows,
        },
        "provenance_case_map.json": {
            "purpose": "audit_only_not_read_by_the_predictor",
            "source_protocol": SOURCE_PROTOCOL,
            "samples": provenance,
        },
        "semantic_evidence.json": {
            "protocol": FROZEN_SEMANTIC_PROTOCOL,
            "samples": semantic_rows,
        },
    }
    _write_json(output / "source_audit.json", source_audit)
    for filename, payload in payloads.items():
        _write_json(output / filename, payload, compact=filename == "semantic_evidence.json")
    frozen_hashes = {filename: _sha256(output / filename) for filename in payloads}
    manifest = {
        "protocol": FROZEN_PROTOCOL,
        "dataset_role": "window_complete_temporal_root_interaction_benchmark",
        "sample_count": len(records),
        "scenario_count": len(scenarios),
        "physical_collection_block_count": len(scenarios) * len(replicates),
        "replicate_index_count": len(replicates),
        "outer_evaluation": {
            "name": "leave_one_replicate_index_out",
            "group_field": "block_index",
        },
        "mechanism_stress_ready": complete_mixed_count == mixed_case_count,
        "cohort_policy": "all planned cases retained; no post-collection filtering",
        "label_inventory": labels,
        "predictor_input_fields": ["case_id", "metric_time_features"],
        "predictor_excluded_fields": source_audit["predictor_excluded"],
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "source_audit_sha256": _sha256(output / "source_audit.json"),
        "frozen_file_sha256": frozen_hashes,
    }
    _write_json(output / "dataset_manifest.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    manifest = freeze(args.source, args.output, args.seed)
    print(json.dumps({key: manifest[key] for key in ("protocol", "sample_count", "scenario_count", "mechanism_stress_ready")}, indent=2))


if __name__ == "__main__":
    main()
