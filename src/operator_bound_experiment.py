"""Audit and evaluate the collected DB-MAGS SQL/operator extension.

The registered 660-case cohort remains the root-cause benchmark.  This module
evaluates the independent post-diagnosis extension: given a detected root, it
ranks the blind SQL candidates and then the operators within each SQL plan.
Ground truth is loaded only by the split builder and scorer; fitted rankers
receive training-fold annotations but never held-out annotations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import shap
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


DEFAULT_DATASET_ROOT = Path("data/dbmags_operator_bound_v4")
DEFAULT_OUTPUT = Path("runs/dbmags-operator-bound-v4/full_report.json")
DEFAULT_SEED = 20260813
ROOTS = (
    "group_by",
    "implicit_conversion",
    "missing_index",
    "order_by",
    "query_whole_table",
    "query_with_too_much_join",
)

_QUERY_NUMERIC_FEATURES = (
    "log_event_count",
    "log_total_elapsed",
    "log_mean_elapsed",
    "log_max_elapsed",
    "log_total_timer_wait",
    "log_total_rows_examined",
    "log_mean_rows_examined",
    "log_total_rows_sent",
    "error_rate",
    "operator_count",
)
_QUERY_CATEGORICAL_FEATURES = (
    "has_full_scan",
    "has_index_scan",
    "has_nested_loop",
    "has_ordering",
    "has_grouping",
    "has_conversion_predicate",
    "has_unrestricted_read",
    "table_order_line",
    "table_orders",
    "table_customer",
    "table_stock",
    "table_district",
    "table_warehouse",
)
_OPERATOR_NUMERIC_FEATURES = (
    "query_log_mean_elapsed",
    "query_log_mean_rows_examined",
    "query_operator_count",
    "plan_depth",
    "log_estimated_rows",
    "log_estimated_output_rows",
    "key_part_count",
    "filtered_fraction",
)
_OPERATOR_CATEGORICAL_FEATURES = (
    "type_all",
    "type_index",
    "type_ref",
    "type_eq_ref",
    "type_nested_loop",
    "type_ordering",
    "type_grouping",
    "has_table",
    "has_key",
    "has_condition",
    "condition_conversion",
    "uses_filesort",
    "uses_temporary_table",
    "table_order_line",
    "table_orders",
    "table_customer",
    "table_stock",
    "table_district",
    "table_warehouse",
)
_OPDIAG_OPERATOR_FEATURES = (
    "plan_depth",
    "log_estimated_rows",
    "log_estimated_output_rows",
    "key_part_count",
    "filtered_fraction",
    "type_all",
    "type_index",
    "type_ref",
    "type_eq_ref",
    "type_nested_loop",
    "type_ordering",
    "type_grouping",
    "has_table",
    "has_key",
    "has_condition",
    "uses_filesort",
    "uses_temporary_table",
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _log1p(value: Any) -> float:
    try:
        numeric = max(float(value or 0.0), 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    return math.log1p(numeric)


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _has_numeric_c_last_predicate(value: Any) -> bool:
    text = _normalise(value).replace("`", "")
    return bool(re.search(r"c_last\s*(?:=|in\s*\(|between\s+)\s*\d", text))


def _operator_key(sql_id: str, operator_id: str) -> str:
    return f"{sql_id}::{operator_id}"


@dataclass(frozen=True)
class BlindOperatorCase:
    case_id: str
    blind: Mapping[str, Any]
    plans: Mapping[str, Any]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class OperatorBoundCase(BlindOperatorCase):
    truth: Mapping[str, Any]

    @property
    def roots(self) -> Tuple[str, ...]:
        return tuple(str(value) for value in self.truth["root_labels"])

    def blind_view(self) -> BlindOperatorCase:
        return BlindOperatorCase(self.case_id, self.blind, self.plans, self.metrics)


def load_cases(dataset_root: Path) -> Tuple[List[OperatorBoundCase], Dict[str, Any]]:
    root = Path(dataset_root)
    manifest = _read_json(root / "dataset_manifest.json")
    manifest_results = list(manifest.get("results") or [])
    if int(manifest.get("sample_schedule_count", 0)) != len(manifest_results):
        raise ValueError("operator-bound collection manifest is incomplete")
    collected_ids = [
        str(row.get("case_id") or "")
        for row in manifest_results
        if row.get("status") == "collected"
    ]
    if not collected_ids or any(not case_id for case_id in collected_ids):
        raise ValueError("operator-bound manifest contains an invalid collected case")
    if len(collected_ids) != len(set(collected_ids)):
        raise ValueError("operator-bound manifest contains duplicate collected cases")
    cases_root = root / "cases"
    case_directories = {path.name for path in cases_root.iterdir() if path.is_dir()}
    if case_directories != set(collected_ids):
        raise ValueError("operator-bound case directories differ from the manifest")
    cases = []
    required = {"blind_candidates.json", "plans.json", "metrics.json", "ground_truth.json"}
    for case_id in sorted(collected_ids):
        path = cases_root / case_id
        present = {item.name for item in path.iterdir() if item.is_file()}
        if present != required:
            raise ValueError(f"collected case is incomplete: {path.name}")
        cases.append(
            OperatorBoundCase(
                case_id=path.name,
                blind=_read_json(path / "blind_candidates.json"),
                plans=_read_json(path / "plans.json"),
                metrics=_read_json(path / "metrics.json"),
                truth=_read_json(path / "ground_truth.json"),
            )
        )
    if not cases:
        raise ValueError("operator-bound dataset contains no collected cases")
    return cases, manifest


def _plan_by_sql_id(case: BlindOperatorCase) -> Dict[str, Mapping[str, Any]]:
    sql_to_id = {str(row["sql"]): str(row["sql_id"]) for row in case.blind["candidates"]}
    return {sql_to_id[sql]: plan for sql, plan in case.plans.items() if sql in sql_to_id}


def _event_metric_mean(candidate: Mapping[str, Any], key: str) -> float:
    values = [
        float((event.get("performance_schema") or {}).get(key) or 0.0)
        for event in candidate.get("events") or []
    ]
    return sum(values) / float(len(values) or 1)


def _independent_mechanism_check(
    case: OperatorBoundCase, root: str
) -> Dict[str, Any]:
    target_sql_id = str(case.truth["target_sql_ids"][root])
    candidates = {
        str(candidate["sql_id"]): candidate for candidate in case.blind["candidates"]
    }
    plans = _plan_by_sql_id(case)
    candidate = candidates[target_sql_id]
    operators = list(plans[target_sql_id].get("operators") or [])
    types = {str(operator.get("operator_type") or "") for operator in operators}
    full_scans = [operator for operator in operators if operator.get("access_type") == "ALL"]
    mean_rows = _event_metric_mean(candidate, "ROWS_EXAMINED")
    evidence: Dict[str, Any]
    passed = False
    if root == "missing_index":
        passed = bool(full_scans) and mean_rows >= 500_000
        evidence = {"has_full_scan": bool(full_scans), "mean_rows_examined": mean_rows}
    elif root == "implicit_conversion":
        control = next(
            row
            for row in case.blind["candidates"]
            if "c_last" in _normalise(row.get("sql"))
            and "'barbarbar'" in _normalise(row.get("sql"))
        )
        control_plan = plans[str(control["sql_id"])]
        target_table = next(
            (operator for operator in operators if operator.get("table_name") == "customer"), {}
        )
        control_table = next(
            (
                operator
                for operator in control_plan.get("operators") or []
                if operator.get("table_name") == "customer"
            ),
            {},
        )
        target_parts = len(target_table.get("used_key_parts") or [])
        control_parts = len(control_table.get("used_key_parts") or [])
        target_rows = float(target_table.get("rows_examined_per_scan") or 0.0)
        control_rows = float(control_table.get("rows_examined_per_scan") or 0.0)
        condition = _normalise(target_table.get("attached_condition"))
        passed = (
            "c_last" in condition
            and target_parts < control_parts
            and target_rows >= max(100.0 * control_rows, 1000.0)
        )
        evidence = {
            "target_used_key_part_count": target_parts,
            "control_used_key_part_count": control_parts,
            "target_estimated_rows": target_rows,
            "control_estimated_rows": control_rows,
            "condition_contains_c_last": "c_last" in condition,
        }
    elif root == "query_with_too_much_join":
        table_count = sum(bool(operator.get("table_name")) for operator in operators)
        passed = "NESTED_LOOP" in types and table_count >= 6 and mean_rows >= 10_000
        evidence = {
            "has_nested_loop": "NESTED_LOOP" in types,
            "plan_table_count": table_count,
            "mean_rows_examined": mean_rows,
        }
    elif root == "order_by":
        ordering = [
            operator for operator in operators
            if operator.get("operator_type") == "ORDERING_OPERATION"
        ]
        passed = (
            bool(ordering)
            and any(bool(operator.get("using_filesort")) for operator in ordering)
            and mean_rows >= 100_000
        )
        evidence = {
            "has_ordering_operator": bool(ordering),
            "uses_filesort": any(bool(operator.get("using_filesort")) for operator in ordering),
            "mean_rows_examined": mean_rows,
        }
    elif root == "group_by":
        grouping = [
            operator for operator in operators
            if operator.get("operator_type") == "GROUPING_OPERATION"
        ]
        passed = (
            bool(grouping)
            and any(bool(operator.get("using_temporary_table")) for operator in grouping)
            and mean_rows >= 1_000_000
        )
        evidence = {
            "has_grouping_operator": bool(grouping),
            "uses_temporary_table": any(
                bool(operator.get("using_temporary_table")) for operator in grouping
            ),
            "mean_rows_examined": mean_rows,
        }
    elif root == "query_whole_table":
        passed = bool(full_scans) and mean_rows >= 1_000_000
        evidence = {"has_full_scan": bool(full_scans), "mean_rows_examined": mean_rows}
    else:
        evidence = {"reason": "unregistered operator-bound root"}
    return {"passed": passed, "evidence": evidence}


def audit_dataset(dataset_root: Path = DEFAULT_DATASET_ROOT) -> Dict[str, Any]:
    cases, manifest = load_cases(dataset_root)
    errors: List[str] = []
    source_protocol = str(manifest.get("protocol") or "")
    if source_protocol != "dbmags-operator-bound-extension-v3":
        errors.append("dataset manifest uses an unexpected protocol")
    phase_counts: Counter[str] = Counter()
    cardinalities: Counter[int] = Counter()
    combinations: Dict[int, set] = defaultdict(set)
    root_counts: Counter[str] = Counter()
    independent_mechanism_passes: Counter[str] = Counter()
    reference_candidate_ids: Dict[str, set] = {}
    variant_counts: Counter[str] = Counter()
    for case in cases:
        cardinalities[len(case.roots)] += 1
        combinations[len(case.roots)].add(tuple(sorted(case.roots)))
        root_counts.update(case.roots)
        candidates = list(case.blind.get("candidates") or [])
        variant = str(case.truth.get("template_variant") or "")
        if not re.fullmatch(r"variant-[1-4]", variant):
            errors.append(f"{case.case_id}: template variant is invalid")
        variant_counts[variant] += 1
        if "template_variant" in case.blind:
            errors.append(f"{case.case_id}: template variant leaked into blind ranker input")
        candidate_sql_ids = {str(row.get("sql_id")) for row in candidates}
        if len(candidates) != 12 or len(candidate_sql_ids) != 12:
            errors.append(f"{case.case_id}: blind candidate pool is not 12 unique SQLs")
        if variant not in reference_candidate_ids:
            reference_candidate_ids[variant] = candidate_sql_ids
        elif candidate_sql_ids != reference_candidate_ids[variant]:
            errors.append(
                f"{case.case_id}: blind SQL candidate inventory differs within {variant}"
            )
        if any(
            candidate.get("plan_status") != "ok" or not candidate.get("events")
            for candidate in candidates
        ):
            errors.append(f"{case.case_id}: candidate plan or execution event is incomplete")
        candidate_operator_keys = {
            _operator_key(str(row.get("sql_id")), str(operator_id))
            for row in candidates
            for operator_id in row.get("operator_ids") or []
        }
        plan_map = _plan_by_sql_id(case)
        plan_operator_keys = {
            _operator_key(sql_id, str(operator["operator_id"]))
            for sql_id, plan in plan_map.items()
            for operator in plan.get("operators") or []
        }
        target_sql_ids = {str(value) for value in case.truth["target_sql_ids"].values()}
        if any(
            not any(event.get("status") == "ok" for event in candidate.get("events") or [])
            for candidate in candidates
            if str(candidate.get("sql_id")) in target_sql_ids
        ):
            errors.append(f"{case.case_id}: a target SQL has no successful execution event")
        target_operator_keys = {
            _operator_key(str(case.truth["target_sql_ids"][root]), str(operator_id))
            for root, operator_ids in case.truth["target_operator_ids_by_root"].items()
            for operator_id in operator_ids
        }
        if not target_sql_ids <= candidate_sql_ids:
            errors.append(f"{case.case_id}: target SQL absent from blind candidate pool")
        if not target_operator_keys <= candidate_operator_keys:
            errors.append(f"{case.case_id}: target operator absent from blind candidate pool")
        if not target_operator_keys <= plan_operator_keys:
            errors.append(f"{case.case_id}: target operator absent from saved plans")
        serialised_blind = json.dumps(case.blind, sort_keys=True).lower()
        forbidden = ("target", "root_labels", "inject", "ground_truth")
        if any(token in serialised_blind for token in forbidden):
            errors.append(f"{case.case_id}: blind candidate payload contains a forbidden key/token")
        timeline = list(case.metrics.get("timeline") or [])
        local_phases = Counter(str(row.get("phase")) for row in timeline)
        phase_counts.update(local_phases)
        if local_phases != Counter({"baseline": 5, "anomaly": 5, "recovery": 5}):
            errors.append(f"{case.case_id}: phase timeline is incomplete")
        if [str(row.get("phase")) for row in timeline] != (
            ["baseline"] * 5 + ["anomaly"] * 5 + ["recovery"] * 5
        ):
            errors.append(f"{case.case_id}: phase timeline order is invalid")
        timestamps = [str(row.get("timestamp") or "") for row in timeline]
        if timestamps != sorted(timestamps):
            errors.append(f"{case.case_id}: phase timestamps are not monotonic")
        if not bool(case.truth.get("operator_evaluable")):
            errors.append(f"{case.case_id}: collected case is not operator evaluable")
        checks = (case.truth.get("validation") or {}).get("mechanism_checks") or {}
        if set(checks) != set(case.roots) or not all(
            bool(row.get("passed")) for row in checks.values()
        ):
            errors.append(f"{case.case_id}: mechanism evidence gate did not pass")
        for root in case.roots:
            check = _independent_mechanism_check(case, root)
            if check["passed"]:
                independent_mechanism_passes[root] += 1
            else:
                errors.append(f"{case.case_id}: independent {root} mechanism audit failed")
    manifest_counts = dict(manifest.get("counts") or {})
    manifest_results = list(manifest.get("results") or [])
    unsupported = [row for row in manifest_results if row.get("status") == "unsupported"]
    if any(
        row.get("status") == "collected" and not bool((row.get("recovery") or {}).get("clean"))
        for row in manifest_results
    ):
        errors.append("a collected manifest row failed its recovery check")
    observed_manifest_counts = {
        status: sum(row.get("status") == status for row in manifest_results)
        for status in sorted({str(row.get("status")) for row in manifest_results})
    }
    if manifest_counts != observed_manifest_counts:
        errors.append("manifest status counts do not match result rows")
    return {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "sample_schedule_count": int(manifest.get("sample_schedule_count", 0)),
        "collected_case_count": len(cases),
        "unsupported_case_count": len(unsupported),
        "manifest_counts": manifest_counts,
        "manifest_status_counts_match": manifest_counts == observed_manifest_counts,
        "uniform_blind_candidate_pool": not any(
            "candidate" in error for error in errors
        ),
        "candidate_pool_scope": "uniform within each registered template variant",
        "collected_by_template_variant": dict(sorted(variant_counts.items())),
        "collected_by_root_cardinality": {str(k): cardinalities[k] for k in sorted(cardinalities)},
        "distinct_root_combinations": {str(k): len(combinations[k]) for k in sorted(combinations)},
        "root_case_counts": {root: root_counts[root] for root in ROOTS},
        "independent_mechanism_pass_counts": {
            root: independent_mechanism_passes[root] for root in ROOTS
        },
        "phase_sample_counts": dict(sorted(phase_counts.items())),
        "unsupported_roots": sorted(
            str(label) for row in unsupported for label in row.get("labels") or []
        ),
        "candidate_truth_separation": (
            "blind_candidates.json contains no target flag or root label; "
            "ground_truth.json is opened only by split construction and scoring"
        ),
        "target_annotation": (
            "isolated annotations generated from preregistered root-specific SQL/plan rules; "
            "not an independent second-annotator adjudication"
        ),
        "operator_identity": "(sql_id, operator_id) composite; plan-node fingerprints are SQL-local",
        "generalization_scope": "held-out root-target SQL templates",
    }


def _case_fold(case: OperatorBoundCase) -> int:
    template_variant = str(case.truth.get("template_variant") or "")
    match = re.fullmatch(r"variant-([1-4])", template_variant)
    if match is None:
        raise ValueError("operator-bound case has no registered template variant")
    return (int(match.group(1)) - 1) % 2


def _tables_and_operators(case: BlindOperatorCase) -> Tuple[Dict[str, List[Mapping[str, Any]]], Dict[str, str]]:
    plans = _plan_by_sql_id(case)
    operators = {sql_id: list(plan.get("operators") or []) for sql_id, plan in plans.items()}
    sql_text = {str(row["sql_id"]): str(row["sql"]) for row in case.blind["candidates"]}
    return operators, sql_text


def _query_features(case: BlindOperatorCase, candidate: Mapping[str, Any]) -> Dict[str, float]:
    sql_id = str(candidate["sql_id"])
    operators, sql_text = _tables_and_operators(case)
    plan_ops = operators.get(sql_id, [])
    events = list(candidate.get("events") or [])
    elapsed = [float(row.get("elapsed_seconds") or 0.0) for row in events]
    perf = [row.get("performance_schema") or {} for row in events]
    rows_examined = [float(row.get("ROWS_EXAMINED") or 0.0) for row in perf]
    rows_sent = [float(row.get("ROWS_SENT") or 0.0) for row in perf]
    timer_wait = [float(row.get("TIMER_WAIT") or 0.0) for row in perf]
    op_types = {_normalise(row.get("operator_type")) for row in plan_ops}
    tables = {_normalise(row.get("table_name")) for row in plan_ops}
    conditions = " ".join(_normalise(row.get("attached_condition")) for row in plan_ops)
    sql = _normalise(sql_text.get(sql_id))
    return {
        "log_event_count": _log1p(len(events)),
        "log_total_elapsed": _log1p(sum(elapsed)),
        "log_mean_elapsed": _log1p(sum(elapsed) / float(len(elapsed) or 1)),
        "log_max_elapsed": _log1p(max(elapsed, default=0.0)),
        "log_total_timer_wait": _log1p(sum(timer_wait)),
        "log_total_rows_examined": _log1p(sum(rows_examined)),
        "log_mean_rows_examined": _log1p(sum(rows_examined) / float(len(rows_examined) or 1)),
        "log_total_rows_sent": _log1p(sum(rows_sent)),
        "error_rate": sum(row.get("status") != "ok" for row in events) / float(len(events) or 1),
        "operator_count": float(len(plan_ops)),
        "has_full_scan": float("all" in op_types),
        "has_index_scan": float(bool(op_types & {"index", "ref", "eq_ref", "range"})),
        "has_nested_loop": float("nested_loop" in op_types),
        "has_ordering": float("ordering_operation" in op_types or "filesort" in op_types),
        "has_grouping": float("grouping_operation" in op_types),
        "has_conversion_predicate": float(
            "cast(" in conditions
            or "convert(" in conditions
            or _has_numeric_c_last_predicate(sql)
        ),
        "has_unrestricted_read": float("select *" in sql and " where " not in f" {sql} "),
        **{f"table_{table}": float(table in tables) for table in (
            "order_line", "orders", "customer", "stock", "district", "warehouse"
        )},
    }


def _operator_features(
    case: BlindOperatorCase,
    candidate: Mapping[str, Any],
    operator: Mapping[str, Any],
) -> Dict[str, float]:
    query = _query_features(case, candidate)
    operator_type = _normalise(operator.get("operator_type"))
    table = _normalise(operator.get("table_name"))
    path = str(operator.get("path") or "")
    condition = _normalise(operator.get("attached_condition"))
    return {
        "query_log_mean_elapsed": query["log_mean_elapsed"],
        "query_log_mean_rows_examined": query["log_mean_rows_examined"],
        "query_operator_count": query["operator_count"],
        "plan_depth": float(path.count(".") + path.count("[")),
        "log_estimated_rows": _log1p(operator.get("rows_examined_per_scan")),
        "log_estimated_output_rows": _log1p(operator.get("rows_produced_per_join")),
        "key_part_count": float(len(operator.get("used_key_parts") or [])),
        "filtered_fraction": float(operator.get("filtered") or 0.0) / 100.0,
        "type_all": float(operator_type == "all"),
        "type_index": float(operator_type == "index"),
        "type_ref": float(operator_type == "ref"),
        "type_eq_ref": float(operator_type == "eq_ref"),
        "type_nested_loop": float(operator_type == "nested_loop"),
        "type_ordering": float(operator_type in {"ordering_operation", "filesort"}),
        "type_grouping": float(operator_type == "grouping_operation"),
        "has_table": float(bool(table)),
        "has_key": float(bool(operator.get("key"))),
        "has_condition": float(bool(condition)),
        "condition_conversion": float(
            "cast(" in condition
            or "convert(" in condition
            or _has_numeric_c_last_predicate(condition)
        ),
        "uses_filesort": float(bool(operator.get("using_filesort"))),
        "uses_temporary_table": float(bool(operator.get("using_temporary_table"))),
        **{f"table_{name}": float(name == table) for name in (
            "order_line", "orders", "customer", "stock", "district", "warehouse"
        )},
    }


def _vector(features: Mapping[str, float], names: Sequence[str]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in names], dtype=np.float64)


def _positive_probability(model: RandomForestClassifier, vectors: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(vectors), dtype=np.float64)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.ones(len(vectors)) if classes == [1] else np.zeros(len(vectors))
    return probabilities[:, classes.index(1)]


def _interpolate_single_row(
    matrix: torch.Tensor, row_index: int, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale one attribution target while keeping its context fixed."""
    target = (matrix[row_index] * float(alpha)).detach().requires_grad_(True)
    selector = torch.zeros((len(matrix), 1), dtype=matrix.dtype, device=matrix.device)
    selector[row_index] = 1.0
    interpolated = matrix.detach() * (1.0 - selector) + target.unsqueeze(0) * selector
    return interpolated, target


def _single_row_ig_scores(
    matrix: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor],
    steps: int = 24,
) -> torch.Tensor:
    """Apply the OpDiag IG path independently to every row of an input matrix."""
    integrals = torch.zeros(len(matrix), dtype=matrix.dtype, device=matrix.device)
    for row_index in range(len(matrix)):
        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            interpolated, target = _interpolate_single_row(matrix, row_index, float(alpha))
            gradient = torch.autograd.grad(objective(interpolated), target)[0].detach()
            integrals[row_index] += torch.linalg.vector_norm(gradient)
    return torch.linalg.vector_norm(matrix, dim=1) * integrals / steps


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(float(value))), float(value))


def _kpi_vector(case: BlindOperatorCase) -> Tuple[Tuple[str, ...], np.ndarray]:
    timeline = list(case.metrics.get("timeline") or [])
    baseline = [row.get("metrics") or {} for row in timeline if row.get("phase") == "baseline"]
    anomaly = [row.get("metrics") or {} for row in timeline if row.get("phase") == "anomaly"]
    if not baseline or not anomaly:
        raise ValueError(f"{case.case_id}: KPI timeline is incomplete")
    names = tuple(sorted(set.intersection(*(set(row) for row in baseline + anomaly))))
    cumulative = {
        "Questions", "Queries", "Com_select", "Com_insert", "Com_update", "Com_delete",
        "Created_tmp_tables", "Created_tmp_disk_tables", "Handler_read_rnd_next",
        "Innodb_row_lock_time",
    }
    values = []
    for name in names:
        if name in cumulative:
            baseline_rate = (float(baseline[-1][name]) - float(baseline[0][name])) / max(
                len(baseline) - 1, 1
            )
            anomaly_rate = (float(anomaly[-1][name]) - float(anomaly[0][name])) / max(
                len(anomaly) - 1, 1
            )
            values.append(_signed_log1p(anomaly_rate - baseline_rate))
        else:
            baseline_mean = sum(float(row[name]) for row in baseline) / len(baseline)
            anomaly_mean = sum(float(row[name]) for row in anomaly) / len(anomaly)
            values.append(_signed_log1p(anomaly_mean - baseline_mean))
    return names, np.asarray(values, dtype=np.float64)


def _plan_feature_matrix(
    case: BlindOperatorCase, candidate: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    sql_id = str(candidate["sql_id"])
    operators, _ = _tables_and_operators(case)
    plan_ops = operators.get(sql_id, [])
    if not plan_ops:
        return np.zeros((1, len(_OPDIAG_OPERATOR_FEATURES)), dtype=np.float64), np.asarray([-1])
    matrix = np.vstack([
        _vector(_operator_features(case, candidate, operator), _OPDIAG_OPERATOR_FEATURES)
        for operator in plan_ops
    ])
    paths = [str(operator.get("path") or "") for operator in plan_ops]
    parents = []
    for index, path in enumerate(paths):
        candidates = [
            (len(parent), parent_index)
            for parent_index, parent in enumerate(paths)
            if parent_index != index and path.startswith(parent + ".")
        ]
        parents.append(max(candidates)[1] if candidates else -1)
    return matrix, np.asarray(parents, dtype=np.int64)


class _TreeKPIModel(nn.Module):
    def __init__(self, operator_width: int, kpi_width: int):
        super().__init__()
        hidden = 24
        self.operator_input = nn.Linear(operator_width, hidden)
        self.operator_self = nn.Linear(hidden, hidden)
        self.operator_children = nn.Linear(hidden, hidden, bias=False)
        self.aggregator = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, kpi_width)
        )

    def encode_plan(self, features: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
        initial = torch.relu(self.operator_input(features))
        child_sum = torch.zeros_like(initial)
        child_count = torch.zeros((len(initial), 1), dtype=initial.dtype, device=initial.device)
        valid = parents >= 0
        if bool(torch.any(valid)):
            child_sum.index_add_(0, parents[valid], initial[valid])
            child_count.index_add_(
                0,
                parents[valid],
                torch.ones((int(valid.sum()), 1), dtype=initial.dtype, device=initial.device),
            )
        child_mean = child_sum / torch.clamp(child_count, min=1.0)
        encoded = torch.relu(self.operator_self(initial) + self.operator_children(child_mean))
        return torch.max(encoded, dim=0).values

    def aggregate(self, embeddings: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        return self.aggregator(torch.sum(embeddings * activity[:, None], dim=0))

    def forward(
        self, plans: Sequence[Tuple[torch.Tensor, torch.Tensor]], activity: torch.Tensor
    ) -> torch.Tensor:
        embeddings = torch.stack([self.encode_plan(features, parents) for features, parents in plans])
        return self.aggregate(embeddings, activity)


class OpDiagDBMAGSReproduction:
    """Paper-structured OpDiag reproduction trained on the MySQL extension."""

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = int(seed)
        self.operator_scaler = StandardScaler()
        self.kpi_scaler = StandardScaler()
        self.kpi_names: Tuple[str, ...] = ()
        self.model: Optional[_TreeKPIModel] = None
        self.root_models: Dict[str, RandomForestClassifier] = {}

    def _plans(
        self, case: BlindOperatorCase, requires_grad: bool = False
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        plans = []
        activity = []
        for candidate in case.blind["candidates"]:
            matrix, parents = _plan_feature_matrix(case, candidate)
            matrix = self.operator_scaler.transform(matrix)
            plans.append((
                torch.tensor(matrix, dtype=torch.float32, requires_grad=requires_grad),
                torch.tensor(parents, dtype=torch.long),
            ))
            activity.append(float(len(candidate.get("events") or [])))
        return plans, torch.tensor(activity, dtype=torch.float32)

    def fit(self, cases: Sequence[OperatorBoundCase]) -> "OpDiagDBMAGSReproduction":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        matrices = [
            _plan_feature_matrix(case, candidate)[0]
            for case in cases
            for candidate in case.blind["candidates"]
        ]
        self.operator_scaler.fit(np.vstack(matrices))
        kpi_rows = []
        for case in cases:
            names, values = _kpi_vector(case)
            if self.kpi_names and names != self.kpi_names:
                raise ValueError("KPI feature inventories differ")
            self.kpi_names = names
            kpi_rows.append(values)
        raw_kpi = np.vstack(kpi_rows)
        standardized_kpi = self.kpi_scaler.fit_transform(raw_kpi)
        standardized_normal = self.kpi_scaler.transform(np.zeros_like(raw_kpi))
        self.model = _TreeKPIModel(len(_OPDIAG_OPERATOR_FEATURES), len(self.kpi_names))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.008, weight_decay=1e-4)
        training_plans = [self._plans(case) for case in cases]
        targets = torch.tensor(standardized_kpi, dtype=torch.float32)
        self.model.train()
        for _ in range(240):
            optimizer.zero_grad()
            predictions = torch.stack([
                self.model(plans, activity)
                for plans, activity in training_plans
            ])
            normal_predictions = torch.stack([
                self.model(plans, torch.zeros_like(activity))
                for plans, activity in training_plans
            ])
            normal_targets = torch.tensor(standardized_normal, dtype=torch.float32)
            loss = torch.mean((predictions - targets) ** 2) + torch.mean(
                (normal_predictions - normal_targets) ** 2
            )
            loss.backward()
            optimizer.step()
        for root_index, root in enumerate(ROOTS):
            labels = np.asarray(
                [root in case.roots for case in cases] + [False] * len(cases),
                dtype=np.int64,
            )
            self.root_models[root] = RandomForestClassifier(
                n_estimators=320,
                max_depth=7,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=self.seed + 500 + root_index,
                n_jobs=1,
            ).fit(np.vstack([standardized_kpi, standardized_normal]), labels)
        self.model.eval()
        return self

    def _kpi_importance(self, case: BlindOperatorCase, root: str) -> np.ndarray:
        _, raw = _kpi_vector(case)
        vector = self.kpi_scaler.transform(raw[None, :])
        model = self.root_models[root]
        explained = shap.TreeExplainer(model).shap_values(vector)
        if isinstance(explained, list):
            explained = explained[-1]
        values = np.asarray(explained, dtype=np.float64)
        if values.ndim == 3:
            values = values[:, :, -1]
        weights = np.abs(values[0])
        if not np.any(weights):
            weights = np.asarray(model.feature_importances_, dtype=np.float64)
        return weights / max(float(np.sum(weights)), 1e-12)

    def rank(self, case: BlindOperatorCase, root: str) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("OpDiag reproduction must be fit before ranking")
        plans, activity = self._plans(case)
        embeddings = torch.stack([
            self.model.encode_plan(features, parents) for features, parents in plans
        ]).detach()
        kpi_weights = torch.tensor(self._kpi_importance(case, root), dtype=torch.float32)
        query_scores = _single_row_ig_scores(
            embeddings,
            lambda interpolated: torch.sum(
                self.model.aggregate(interpolated, activity) * kpi_weights
            ),
        )
        candidates = list(case.blind["candidates"])
        sql_rows = sorted(
            (
                {"sql_id": str(candidate["sql_id"]), "score": float(score)}
                for candidate, score in zip(candidates, query_scores)
            ),
            key=lambda row: (-row["score"], row["sql_id"]),
        )
        operators_by_sql, _ = _tables_and_operators(case)
        operator_rankings: Dict[str, List[Dict[str, Any]]] = {}
        for candidate in candidates:
            sql_id = str(candidate["sql_id"])
            matrix, parents = _plan_feature_matrix(case, candidate)
            standardized = self.operator_scaler.transform(matrix)
            source = torch.tensor(standardized, dtype=torch.float32)
            parent_tensor = torch.tensor(parents, dtype=torch.long)
            scores = _single_row_ig_scores(
                source,
                lambda interpolated: torch.linalg.vector_norm(
                    self.model.encode_plan(interpolated, parent_tensor)
                ),
            )
            plan_ops = operators_by_sql.get(sql_id, [])
            operator_rankings[sql_id] = sorted(
                (
                    {
                        "operator_key": _operator_key(sql_id, str(operator["operator_id"])),
                        "operator_id": str(operator["operator_id"]),
                        "score": float(score),
                    }
                    for operator, score in zip(plan_ops, scores)
                ),
                key=lambda row: (-row["score"], row["operator_key"]),
            )
        return {"sql_ranking": sql_rows, "operator_rankings_by_sql": operator_rankings}

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "OpDiag-DBMAGS reproduction",
            "native_official_implementation": False,
            "paper_alignment": (
                "schema-independent plan-operator features, tree convolution and max pooling, "
                "concurrent-query sum aggregator trained to predict KPIs, independent root random "
                "forests, and hierarchical integrated-gradient query/operator attribution"
            ),
            "training_sql_or_operator_targets_used": False,
            "integrated_gradient_path": (
                "only the attributed query or operator is scaled from zero to its observed "
                "value; all concurrent context remains fixed"
            ),
            "normal_training_control": "per-case zero-activity baseline with zero KPI delta",
            "grouped_query_activity": (
                "each sampled SQL template encoding is multiplied by its observed execution "
                "count, equivalent to summing the expanded concurrent query instances"
            ),
            "test_ground_truth_read_by_ranker": False,
            "kpi_features": list(self.kpi_names),
            "operator_features": list(_OPDIAG_OPERATOR_FEATURES),
        }


class LearnedHierarchicalRanker:
    """EPDG-constrained root-specific SQL and operator ranker."""

    def __init__(self, seed: int = DEFAULT_SEED, *, root_conditioning: bool = True):
        self.seed = int(seed)
        self.root_conditioning = bool(root_conditioning)
        self.query_models: Dict[str, RandomForestClassifier] = {}
        self.operator_models: Dict[str, RandomForestClassifier] = {}
        self.query_names: Tuple[str, ...] = ()
        self.operator_names: Tuple[str, ...] = ()

    def fit(self, cases: Sequence[OperatorBoundCase]) -> "LearnedHierarchicalRanker":
        self.query_names = _QUERY_NUMERIC_FEATURES + _QUERY_CATEGORICAL_FEATURES
        self.operator_names = _OPERATOR_NUMERIC_FEATURES + _OPERATOR_CATEGORICAL_FEATURES
        model_targets: Tuple[Tuple[str, Optional[str]], ...] = (
            tuple((root, root) for root in ROOTS)
            if self.root_conditioning else
            (("shared", None),)
        )
        for root_index, (model_key, root) in enumerate(model_targets):
            query_x: List[np.ndarray] = []
            query_y: List[int] = []
            operator_x: List[np.ndarray] = []
            operator_y: List[int] = []
            for case in cases:
                target_sqls = (
                    {str(case.truth["target_sql_ids"].get(root) or "")}
                    if root is not None else
                    {str(value) for value in case.truth["target_sql_ids"].values()}
                )
                target_sqls.discard("")
                target_operators = {
                    _operator_key(str(case.truth["target_sql_ids"][label]), str(value))
                    for label, values in case.truth["target_operator_ids_by_root"].items()
                    if root is None or label == root
                    for value in values
                }
                operators, _ = _tables_and_operators(case)
                for candidate in case.blind["candidates"]:
                    sql_id = str(candidate["sql_id"])
                    query_x.append(_vector(_query_features(case, candidate), self.query_names))
                    query_y.append(int(sql_id in target_sqls))
                    for operator in operators.get(sql_id, []):
                        operator_x.append(
                            _vector(_operator_features(case, candidate, operator), self.operator_names)
                        )
                        operator_y.append(
                            int(_operator_key(sql_id, str(operator["operator_id"])) in target_operators)
                        )
            if len(set(query_y)) < 2 or len(set(operator_y)) < 2:
                raise ValueError(f"training fold lacks positive/negative examples for {model_key}")
            self.query_models[model_key] = RandomForestClassifier(
                n_estimators=320,
                max_depth=7,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=self.seed + root_index,
                n_jobs=1,
            ).fit(np.vstack(query_x), np.asarray(query_y, dtype=np.int64))
            self.operator_models[model_key] = RandomForestClassifier(
                n_estimators=320,
                max_depth=7,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=self.seed + 100 + root_index,
                n_jobs=1,
            ).fit(np.vstack(operator_x), np.asarray(operator_y, dtype=np.int64))
        return self

    def rank(self, case: BlindOperatorCase, root: str) -> Dict[str, Any]:
        model_key = root if self.root_conditioning else "shared"
        candidates = list(case.blind["candidates"])
        query_vectors = np.vstack([
            _vector(_query_features(case, row), self.query_names) for row in candidates
        ])
        query_scores = _positive_probability(self.query_models[model_key], query_vectors)
        sql_rows = sorted(
            (
                {"sql_id": str(row["sql_id"]), "score": float(score)}
                for row, score in zip(candidates, query_scores)
            ),
            key=lambda row: (-row["score"], row["sql_id"]),
        )
        operators, _ = _tables_and_operators(case)
        operator_rankings: Dict[str, List[Dict[str, Any]]] = {}
        candidate_by_id = {str(row["sql_id"]): row for row in candidates}
        for sql_row in sql_rows:
            sql_id = sql_row["sql_id"]
            plan_ops = operators.get(sql_id, [])
            if not plan_ops:
                continue
            vectors = np.vstack([
                _vector(
                    _operator_features(case, candidate_by_id[sql_id], operator),
                    self.operator_names,
                )
                for operator in plan_ops
            ])
            local_scores = _positive_probability(self.operator_models[model_key], vectors)
            operator_rankings[sql_id] = sorted(({
                    "operator_key": _operator_key(sql_id, str(operator["operator_id"])),
                    "sql_id": sql_id,
                    "operator_id": str(operator["operator_id"]),
                    "score": float(local_score),
                } for operator, local_score in zip(plan_ops, local_scores)),
                key=lambda row: (-row["score"], row["operator_key"]),
            )
        return {"sql_ranking": sql_rows, "operator_rankings_by_sql": operator_rankings}

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": (
                "HyperDBDiag EPDG binder"
                if self.root_conditioning else
                "root-unconditioned SQL/operator binder"
            ),
            "hierarchy": (
                "detected root -> blind SQL candidates -> operators within ranked SQLs"
                if self.root_conditioning else
                "blind SQL candidates -> operators within ranked SQLs"
            ),
            "training": (
                "root-specific random forests fitted only on the opposite registered training fold"
                if self.root_conditioning else
                "one shared random-forest pair fitted only on the opposite registered training fold"
            ),
            "query_features": list(self.query_names),
            "operator_features": list(self.operator_names),
            "test_ground_truth_read_by_ranker": False,
            "epdg_binding": (
                "root-specific SQL/plan/operator path evidence"
                if self.root_conditioning else
                "not used"
            ),
        }


def _rank_of(ranking: Sequence[Mapping[str, Any]], key: str, targets: Iterable[str]) -> Optional[int]:
    target_set = set(targets)
    for index, row in enumerate(ranking, start=1):
        if str(row[key]) in target_set:
            return index
    return None


def _hit_summary(hits: Sequence[bool]) -> Dict[str, Any]:
    materialized = [bool(value) for value in hits]
    return {
        "denominator": len(materialized),
        "hit_at_1": sum(materialized) / float(len(materialized) or 1),
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    case_rows = {str(row["case_id"]): row for row in rows}
    summary = {
        "root_sql_pairs": _hit_summary([row["sql_hit_at_1"] for row in rows]),
        "root_operator_pairs": _hit_summary([row["operator_hit_at_1"] for row in rows]),
        "case_count": len(case_rows),
        "case_full_sql_coverage_at_1": sum(row["all_sql_hit_at_1"] for row in case_rows.values())
        / float(len(case_rows) or 1),
        "case_full_chain_coverage_at_1": sum(row["all_chain_hit_at_1"] for row in case_rows.values())
        / float(len(case_rows) or 1),
    }
    by_cardinality = {}
    for cardinality in sorted({int(row["root_cardinality"]) for row in rows}):
        selected = [row for row in rows if int(row["root_cardinality"]) == cardinality]
        selected_cases = {str(row["case_id"]): row for row in selected}
        by_cardinality[str(cardinality)] = {
            "case_count": len(selected_cases),
            "root_sql_pairs": _hit_summary([row["sql_hit_at_1"] for row in selected]),
            "root_operator_pairs": _hit_summary([row["operator_hit_at_1"] for row in selected]),
            "case_full_sql_coverage_at_1": sum(
                row["all_sql_hit_at_1"] for row in selected_cases.values()
            ) / float(len(selected_cases) or 1),
            "case_full_chain_coverage_at_1": sum(
                row["all_chain_hit_at_1"] for row in selected_cases.values()
            ) / float(len(selected_cases) or 1),
        }
    summary["by_root_cardinality"] = by_cardinality
    summary["by_root"] = {
        root: {
            "case_count": sum(row["root"] == root for row in rows),
            "sql": _hit_summary([row["sql_hit_at_1"] for row in rows if row["root"] == root]),
            "operator": _hit_summary([
                row["operator_hit_at_1"] for row in rows if row["root"] == root
            ]),
        }
        for root in ROOTS
    }
    return summary


def _evaluate(
    method: str,
    cases: Sequence[OperatorBoundCase],
    seed: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    fold_metadata = []
    for fold in (0, 1):
        training = [case for case in cases if _case_fold(case) != fold]
        evaluation = [case for case in cases if _case_fold(case) == fold]
        if method == "opdiag_reproduction":
            ranker = OpDiagDBMAGSReproduction(seed + fold).fit(training)
        elif method in {"unconditioned_binder", "hyperdbdiag"}:
            ranker = LearnedHierarchicalRanker(
                seed + fold,
                root_conditioning=method == "hyperdbdiag",
            ).fit(training)
        else:
            raise ValueError("unknown evaluation method")
        fold_metadata.append({
            "fold": fold,
            "training_case_count": len(training),
            "evaluation_case_count": len(evaluation),
            "model": ranker.metadata(),
        })
        for case in evaluation:
            blind_case = case.blind_view()
            local_rows = []
            for root in case.roots:
                ranked = ranker.rank(blind_case, root)
                target_sql = str(case.truth["target_sql_ids"][root])
                target_operators = {
                    _operator_key(target_sql, str(value))
                    for value in case.truth["target_operator_ids_by_root"][root]
                }
                target_operator_ranking = ranked["operator_rankings_by_sql"].get(
                    target_sql, []
                )
                sql_hit_at_1 = _rank_of(
                    ranked["sql_ranking"], "sql_id", {target_sql}
                ) == 1
                operator_hit_at_1 = _rank_of(
                    target_operator_ranking, "operator_key", target_operators
                ) == 1
                local_rows.append({
                    "case_id": case.case_id,
                    "fold": fold,
                    "root_cardinality": len(case.roots),
                    "root": root,
                    "sql_hit_at_1": sql_hit_at_1,
                    "operator_hit_at_1": operator_hit_at_1,
                    "top_sql_ids": [row["sql_id"] for row in ranked["sql_ranking"][:1]],
                    "top_operator_keys": [
                        row["operator_key"] for row in target_operator_ranking[:1]
                    ],
                })
            sql_full = all(row["sql_hit_at_1"] for row in local_rows)
            chain_full = all(
                row["sql_hit_at_1"] and row["operator_hit_at_1"]
                for row in local_rows
            )
            for row in local_rows:
                row["all_sql_hit_at_1"] = sql_full
                row["all_chain_hit_at_1"] = chain_full
            rows.extend(local_rows)
    return {
        "stage": "conditional_post_diagnosis",
        "root_input": "oracle detected root, matching OpDiag's post-detection attribution stage",
        "folds": fold_metadata,
        "metrics": _summarize_rows(rows),
        "results": rows,
    }


def run(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    cases, manifest = load_cases(dataset_root)
    audit = audit_dataset(dataset_root)
    if audit["status"] != "valid":
        raise ValueError("operator-bound dataset failed integrity audit")
    opdiag = _evaluate("opdiag_reproduction", cases, seed)
    unconditioned = _evaluate("unconditioned_binder", cases, seed)
    hyperdbdiag = _evaluate("hyperdbdiag", cases, seed)

    def hit_at_1(configuration: Mapping[str, Any]) -> Dict[str, float]:
        metrics = configuration["metrics"]
        return {
            "sql_hit_at_1": float(metrics["root_sql_pairs"]["hit_at_1"]),
            "operator_hit_at_1": float(metrics["root_operator_pairs"]["hit_at_1"]),
        }

    report = {
        "protocol": {
            "name": "dbmags-operator-bound-post-diagnosis-v1",
            "seed": int(seed),
            "source_protocol": str(manifest.get("protocol")),
            "dataset_role": str(manifest.get("dataset_role")),
            "split": (
                "two-fold root-target SQL-template holdout (variants 1/3 versus 2/4), "
                "balanced within every root combination"
            ),
            "generalization_scope": (
                "held-out target SQL templates within the same schema; shared controls remain"
            ),
            "scope": (
                "SQL/operator attribution after root detection; this 5-second extension is not "
                "merged with the 75-second 660-case root benchmark"
            ),
            "metrics": "root-SQL and root-operator Hit@1, and per-case full coverage@1",
            "operator_metric_scope": (
                "Operator Hit@1 is conditional on the ground-truth SQL plan, matching OpDiag's "
                "hierarchical operator-ranking stage; it is not an end-to-end operator metric"
            ),
            "chain_metric_scope": (
                "per-case full chain coverage@1 requires every root's target SQL and its target "
                "operator to both rank first in the corresponding hierarchy"
            ),
            "ablation_mapping_scope": (
                "root detection and conditional fine-grained attribution use separate datasets; "
                "variants without EPDG use the shared unconditioned binder, while variants with "
                "EPDG use the root-conditioned binder. Local/LLM arbitration changes the upstream "
                "root set but not oracle-root conditional SQL/operator ranking."
            ),
        },
        "audit": audit,
        "methods": {
            "opdiag_dbmags_reproduction": opdiag,
            "hyperdbdiag": hyperdbdiag,
        },
        "attribution_configurations": {
            "without_epdg_root_conditioning": unconditioned,
            "epdg_root_conditioned": hyperdbdiag,
        },
        "ablation_variant_hit_at_1": {
            "ordinary_binary_graph": hit_at_1(unconditioned),
            "hypergraph_without_epdg": hit_at_1(unconditioned),
            "epdg_grounded_hypergraph": hit_at_1(hyperdbdiag),
            "epdg_local_judge": hit_at_1(hyperdbdiag),
            "hypergraph_llm_arbitration": hit_at_1(hyperdbdiag),
            "hyperdbdiag": hit_at_1(hyperdbdiag),
        },
        "end_to_end_status": {
            "status": "not_evaluated_on_this_extension",
            "reason": (
                "the extension has 5-second baseline/anomaly/recovery windows and cannot be fed "
                "to the registered 75-second root classifiers without changing their input contract"
            ),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if args.audit_only:
        report = audit_dataset(args.dataset_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "valid" else 1
    report = run(args.dataset_root, args.output, args.seed)
    compact = {
        name: {
            "sql_hit_at_1": row["metrics"]["root_sql_pairs"]["hit_at_1"],
            "operator_hit_at_1": row["metrics"]["root_operator_pairs"]["hit_at_1"],
        }
        for name, row in report["methods"].items()
    }
    print(json.dumps({"output": str(args.output), "methods": compact}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
