"""Validated reader for the immutable metric-only DB-MAGS artifact."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from metric_frozen_schema import FEATURE_SCHEMA, FEATURE_SCHEMA_SHA256, FROZEN_PROTOCOL


OPAQUE_CASE_ID = re.compile(r"case-[0-9]{4,}")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenMetricDataset:
    case_ids: Tuple[str, ...]
    features: Mapping[str, np.ndarray]
    labels_by_case: Mapping[str, Tuple[str, ...]]
    labels: Tuple[str, ...]
    replicate_by_case: Mapping[str, int]
    scenario_by_case: Mapping[str, str]
    scenario_count: int
    physical_collection_block_count: int
    replicate_count: int

    @property
    def feature_count(self) -> int:
        return int(FEATURE_SCHEMA["feature_count"])


def load_frozen_metric_dataset(root: Path) -> FrozenMetricDataset:
    root = Path(root)
    manifest = _read_json(root / "dataset_manifest.json")
    if manifest.get("protocol") != FROZEN_PROTOCOL:
        raise ValueError("unexpected frozen metric protocol")
    if manifest.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
        raise ValueError("frozen metric feature schema is not registered")
    if manifest.get("outer_evaluation") != {
        "name": "leave_one_replicate_index_out",
        "group_field": "block_index",
    }:
        raise ValueError("frozen metric outer evaluation is not registered")

    frozen_hashes = dict(manifest.get("frozen_file_sha256") or {})
    required_files = {
        "frozen_inputs.json",
        "fold_manifest.json",
        "ground_truth.json",
        "provenance_case_map.json",
        "semantic_evidence.json",
    }
    if set(frozen_hashes) != required_files:
        raise ValueError("frozen metric file inventory is incomplete")
    for filename, digest in frozen_hashes.items():
        if _sha256(root / filename) != digest:
            raise ValueError(f"frozen metric file hash mismatch: {filename}")
    source_audit_sha256 = str(manifest.get("source_audit_sha256") or "")
    if _sha256(root / "source_audit.json") != source_audit_sha256:
        raise ValueError("frozen metric source audit hash mismatch")

    inputs = _read_json(root / "frozen_inputs.json")
    folds = _read_json(root / "fold_manifest.json")
    truth = _read_json(root / "ground_truth.json")
    provenance = _read_json(root / "provenance_case_map.json")
    if any(payload.get("protocol") != FROZEN_PROTOCOL for payload in (inputs, folds, truth)):
        raise ValueError("frozen metric artifacts use different protocols")
    if set(inputs) != {"protocol", "feature_schema", "feature_schema_sha256", "samples"}:
        raise ValueError("predictor input contains an unregistered top-level field")
    if inputs.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("predictor input feature schema differs from the registry")
    if set(folds) != {"protocol", "inner_oof_group", "samples"}:
        raise ValueError("fold manifest contains an unregistered top-level field")
    if folds.get("inner_oof_group") != {
        "name": "replicate_index_cross_fit",
        "group_field": "block_index",
        "purpose": "training_fold_h_residual_construction_only",
    }:
        raise ValueError("frozen metric inner OOF group is not registered")
    if provenance.get("purpose") != "audit_only_not_read_by_the_predictor":
        raise ValueError("provenance is not marked audit-only")

    features: Dict[str, np.ndarray] = {}
    for row in inputs.get("samples") or []:
        if set(row) != {"case_id", "metric_time_features"}:
            raise ValueError("predictor sample contains an unregistered field")
        case_id = str(row["case_id"])
        values = np.asarray(row["metric_time_features"], dtype=np.float64)
        if (
            not OPAQUE_CASE_ID.fullmatch(case_id)
            or case_id in features
            or values.shape != (int(FEATURE_SCHEMA["feature_count"]),)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("predictor sample is malformed")
        features[case_id] = values

    replicate_by_case: Dict[str, int] = {}
    for row in folds.get("samples") or []:
        if set(row) != {"case_id", "block_index"}:
            raise ValueError("fold sample contains an unregistered field")
        case_id = str(row["case_id"])
        replicate = int(row["block_index"])
        if case_id in replicate_by_case or replicate < 1:
            raise ValueError("fold sample is malformed")
        replicate_by_case[case_id] = replicate

    labels = tuple(str(value) for value in truth.get("label_inventory") or [])
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("ground truth label inventory is malformed")
    labels_by_case: Dict[str, Tuple[str, ...]] = {}
    for row in truth.get("samples") or []:
        if set(row) != {"case_id", "expected_labels"}:
            raise ValueError("ground-truth sample contains an unregistered field")
        case_id = str(row["case_id"])
        expected = tuple(str(value) for value in row["expected_labels"])
        if case_id in labels_by_case or not expected or not set(expected) <= set(labels):
            raise ValueError("ground-truth sample is malformed")
        labels_by_case[case_id] = expected

    scenario_by_case: Dict[str, str] = {}
    table_by_case: Dict[str, str] = {}
    source_case_by_case: Dict[str, str] = {}
    provenance_fields = {
        "case_id",
        "source_case_id",
        "source_block_id",
        "source_replicate_index",
        "source_scenario",
        "source_table",
    }
    for row in provenance.get("samples") or []:
        if set(row) != provenance_fields:
            raise ValueError("provenance sample contains an unregistered field")
        case_id = str(row["case_id"])
        scenario = str(row["source_scenario"])
        source_case_id = str(row["source_case_id"])
        if case_id in scenario_by_case or not scenario or not source_case_id:
            raise ValueError("provenance sample is malformed")
        if int(row["source_replicate_index"]) != replicate_by_case.get(case_id):
            raise ValueError("provenance and fold replicate indexes differ")
        scenario_by_case[case_id] = scenario
        table_by_case[case_id] = str(row["source_table"])
        source_case_by_case[case_id] = source_case_id

    case_ids = tuple(features)
    inventories = (
        set(replicate_by_case),
        set(labels_by_case),
        set(scenario_by_case),
        set(table_by_case),
        set(source_case_by_case),
    )
    if not case_ids or any(set(case_ids) != inventory for inventory in inventories):
        raise ValueError("frozen metric artifact case inventories differ")

    sample_count = int(manifest.get("sample_count", 0))
    scenario_count = int(manifest.get("scenario_count", 0))
    physical_block_count = int(manifest.get("physical_collection_block_count", 0))
    replicate_count = int(manifest.get("replicate_index_count", 0))
    replicate_counts = Counter(replicate_by_case.values())
    scenario_replicate_counts = Counter(
        (scenario_by_case[case_id], replicate_by_case[case_id]) for case_id in case_ids
    )
    if sample_count != len(case_ids) or scenario_count != len(set(scenario_by_case.values())):
        raise ValueError("frozen metric dataset counts are inconsistent")
    if set(replicate_by_case.values()) != set(range(1, replicate_count + 1)):
        raise ValueError("frozen metric replicate inventory is incomplete")
    if physical_block_count != scenario_count * replicate_count:
        raise ValueError("frozen metric physical block count is inconsistent")
    if set(replicate_counts.values()) != {sample_count // replicate_count}:
        raise ValueError("frozen metric replicate folds are unbalanced")
    if set(scenario_replicate_counts.values()) != {5}:
        raise ValueError("each scenario replicate must contain five planned conditions")

    return FrozenMetricDataset(
        case_ids=case_ids,
        features=features,
        labels_by_case=labels_by_case,
        labels=labels,
        replicate_by_case=replicate_by_case,
        scenario_by_case=scenario_by_case,
        scenario_count=scenario_count,
        physical_collection_block_count=physical_block_count,
        replicate_count=replicate_count,
    )
