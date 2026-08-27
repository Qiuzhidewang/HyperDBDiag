"""Run the DB-MAGS main comparison on the registered six outer folds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from dbaiops_baseline import DBAIOpsRootCauseClassifier
from hypergraph_core import _component_f1, _exact
from metric_frozen_dataset import load_frozen_metric_dataset
from metric_frozen_schema import (
    FEATURE_METRICS,
    FEATURE_SCHEMA_SHA256,
    FROZEN_PROTOCOL,
    RELATIVE_TIME_BINS_SECONDS,
)
from opdiag_baseline import OpDiagRootCauseClassifier


DEFAULT_DBMAGS_ROOT = Path("data/dbmags_interaction_v10_metric_only")
DEFAULT_ABLATION_REPORT = Path("runs/dbmags-ablation/full_report_llm_xhigh.json")
DEFAULT_OUTPUT = Path("runs/dbmags-main-comparison/full_report_llm_xhigh.json")
DEFAULT_SEED = 20260802
OPDIAG_SEED = 42
CANONICAL_METHODS: Mapping[str, Mapping[str, str]] = {
    "hyperdbdiag": {
        "ablation_method": "hyperdbdiag",
        "display_name": "HyperDBDiag",
        "description": "Complete EPDG-grounded HyperDBDiag pipeline.",
    },
    "epdg_hypergraph": {
        "ablation_method": "hypergraph",
        "display_name": "EPDG + Hypergraph",
        "description": "EPDG-grounded signed-incidence hypergraph before LLM arbitration.",
    },
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / float(len(materialized) or 1)


def _case_rows(
    frozen: Any,
    case_ids: Sequence[str],
    predicted: Sequence[Sequence[str]],
    details: Sequence[Mapping[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    rows = []
    for index, (case_id, prediction) in enumerate(zip(case_ids, predicted)):
        replicate = int(frozen.replicate_by_case[case_id])
        scenario = str(frozen.scenario_by_case[case_id])
        row = {
            "case_id": case_id,
            "block_id": f"replicate-{replicate:02d}:{scenario}",
            "scenario": scenario,
            "expected_labels": list(frozen.labels_by_case[case_id]),
            "predicted_labels": list(prediction),
        }
        if details is not None:
            row["diagnostics"] = dict(details[index])
        rows.append(row)
    return rows


def _summary(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = [list(row["expected_labels"]) for row in rows]
    predicted = [list(row["predicted_labels"]) for row in rows]
    by_cardinality = {}
    for cardinality in sorted({len(row) for row in expected}):
        indices = [index for index, row in enumerate(expected) if len(row) == cardinality]
        by_cardinality[str(cardinality)] = {
            "sample_count": len(indices),
            "exact_set_accuracy": _exact(
                [expected[index] for index in indices],
                [predicted[index] for index in indices],
            ),
            "component_f1": _component_f1(
                [expected[index] for index in indices],
                [predicted[index] for index in indices],
                labels,
            ),
        }
    scenario_indices: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        scenario_indices[str(row["scenario"])].append(index)
    true_positive = sum(
        label in actual and label in observed
        for actual, observed in zip(expected, predicted)
        for label in labels
    )
    false_positive = sum(
        label not in actual and label in observed
        for actual, observed in zip(expected, predicted)
        for label in labels
    )
    false_negative = sum(
        label in actual and label not in observed
        for actual, observed in zip(expected, predicted)
        for label in labels
    )
    precision = true_positive / float(true_positive + false_positive or 1)
    recall = true_positive / float(true_positive + false_negative or 1)
    return {
        "overall": {
            "sample_count": len(rows),
            "exact_set_accuracy": _exact(expected, predicted),
            "component_precision": precision,
            "component_recall": recall,
            "component_f1": _component_f1(expected, predicted, labels),
            "full_root_coverage": _mean(
                set(actual) <= set(observed)
                for actual, observed in zip(expected, predicted)
            ),
            "mean_predicted_roots": _mean(len(row) for row in predicted),
        },
        "by_root_cardinality": by_cardinality,
        "by_scenario": {
            scenario: {
                "sample_count": len(indices),
                "exact_set_accuracy": _exact(
                    [expected[index] for index in indices],
                    [predicted[index] for index in indices],
                ),
            }
            for scenario, indices in sorted(scenario_indices.items())
        },
        "method_metadata": dict(metadata),
        "results": list(rows),
    }


def _paired_bootstrap(
    baseline_rows: Sequence[Mapping[str, Any]],
    challenger_rows: Sequence[Mapping[str, Any]],
    seed: int,
    repetitions: int,
) -> Dict[str, Any]:
    baseline = {str(row["case_id"]): row for row in baseline_rows}
    challenger = {str(row["case_id"]): row for row in challenger_rows}
    if set(baseline) != set(challenger):
        raise ValueError("paired methods cover different cases")
    blocks: Dict[str, List[str]] = defaultdict(list)
    for case_id, row in baseline.items():
        blocks[str(row["block_id"])].append(case_id)
    block_ids = [case_ids for _, case_ids in sorted(blocks.items())]

    def correctness(rows: Mapping[str, Mapping[str, Any]]) -> List[np.ndarray]:
        return [
            np.asarray(
                [
                    set(rows[case_id]["expected_labels"])
                    == set(rows[case_id]["predicted_labels"])
                    for case_id in case_ids
                ],
                dtype=np.float64,
            )
            for case_ids in block_ids
        ]

    baseline_values = correctness(baseline)
    challenger_values = correctness(challenger)
    point = float(
        np.mean(np.concatenate(challenger_values))
        - np.mean(np.concatenate(baseline_values))
    )
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(block_ids), size=len(block_ids))
        samples.append(
            float(
                np.mean(np.concatenate([challenger_values[index] for index in selected]))
                - np.mean(np.concatenate([baseline_values[index] for index in selected]))
            )
        )
    return {
        "metric": "exact_set_accuracy",
        "unit": "DB-MAGS scenario-by-replicate block with five scheduled conditions",
        "block_count": len(block_ids),
        "repetitions": repetitions,
        "point_estimate": point,
        "confidence_interval_95": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
    }


def _canonical_method(
    report: Mapping[str, Any], seed: int, method_name: str
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if int(report["protocol"]["seed"]) != int(seed):
        raise RuntimeError("main comparison and ablation seeds differ")
    dataset = report["datasets"]["dbmags_sql_interaction_subset"]
    method = dataset["methods"][method_name]
    if method.get("stage", {}).get("estimand_status") != "evaluated":
        raise RuntimeError(f"ablation method {method_name} was not evaluated")
    return method, dataset


def run(
    dbmags_root: Path = DEFAULT_DBMAGS_ROOT,
    ablation_report_path: Path = DEFAULT_ABLATION_REPORT,
    output_path: Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
    bootstrap_repetitions: int = 5000,
    canonical_method: str = "hyperdbdiag",
) -> Dict[str, Any]:
    if canonical_method not in CANONICAL_METHODS:
        raise ValueError("unsupported canonical method")
    if bootstrap_repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    frozen = load_frozen_metric_dataset(dbmags_root)
    report = json.loads(ablation_report_path.read_text(encoding="utf-8"))
    specification = CANONICAL_METHODS[canonical_method]
    canonical, canonical_dataset = _canonical_method(
        report, seed, specification["ablation_method"]
    )
    fold_metadata = {
        str(row["outer_split"]): row for row in canonical_dataset["folds"]
    }
    labels = tuple(frozen.labels)
    feature_names = tuple(
        f"{metric}[{start:g}-{end:g}s]"
        for start, end in RELATIVE_TIME_BINS_SECONDS
        for metric in FEATURE_METRICS
    )
    rows_by_method: Dict[str, List[Dict[str, Any]]] = {
        "opdiag": [],
        "dbaiops": [],
    }
    folds = []
    for held_out_replicate in range(1, frozen.replicate_count + 1):
        split_id = f"leave_replicate_index_out:{held_out_replicate}"
        train_ids = tuple(
            case_id
            for case_id in frozen.case_ids
            if frozen.replicate_by_case[case_id] != held_out_replicate
        )
        eval_ids = tuple(
            case_id
            for case_id in frozen.case_ids
            if frozen.replicate_by_case[case_id] == held_out_replicate
        )
        train_x = np.asarray([frozen.features[case_id] for case_id in train_ids])
        eval_x = np.asarray([frozen.features[case_id] for case_id in eval_ids])
        train_labels = [frozen.labels_by_case[case_id] for case_id in train_ids]
        train_groups = [frozen.replicate_by_case[case_id] for case_id in train_ids]
        opdiag = OpDiagRootCauseClassifier(seed=OPDIAG_SEED).fit(
            train_x, train_labels, labels
        )
        opdiag_details = opdiag.predict_details(eval_x)
        opdiag_prediction = [row["predicted_labels"] for row in opdiag_details]
        rows_by_method["opdiag"].extend(
            _case_rows(frozen, eval_ids, opdiag_prediction, opdiag_details)
        )
        dbaiops = DBAIOpsRootCauseClassifier(
            seed=OPDIAG_SEED, metric_count=len(FEATURE_METRICS)
        ).fit(train_x, train_labels, labels, feature_names)
        dbaiops_details = dbaiops.predict_details(eval_x)
        dbaiops_prediction = [row["predicted_labels"] for row in dbaiops_details]
        rows_by_method["dbaiops"].extend(
            _case_rows(frozen, eval_ids, dbaiops_prediction, dbaiops_details)
        )
        fold_canonical_metadata = fold_metadata[split_id].get(
            specification["ablation_method"]
        )
        if fold_canonical_metadata is None:
            # LLM-backed rows keep their detailed calibration audit at the
            # dataset-method level; deterministic rows retain the historical
            # per-fold model metadata. Preserve both report contracts.
            fold_canonical_metadata = {
                "display_name": specification["display_name"],
                "stage": canonical.get("stage", {}),
                "metadata_scope": "dataset_method_stage",
            }
        folds.append(
            {
                "outer_split": split_id,
                "train_count": len(train_ids),
                "evaluation_count": len(eval_ids),
                "train_evaluation_id_overlap": sorted(set(train_ids) & set(eval_ids)),
                "inner_oof_group_count": len(set(train_groups)),
                "opdiag": opdiag.training_metadata,
                "dbaiops": dbaiops.training_metadata,
                "canonical_method": fold_canonical_metadata,
            }
        )
    canonical_by_case = {
        str(row["case_id"]): row for row in canonical["results"]
    }
    if set(canonical_by_case) != set(frozen.case_ids):
        raise RuntimeError("canonical ablation predictions do not cover the frozen cohort")
    rows_by_method[canonical_method] = _case_rows(
        frozen,
        frozen.case_ids,
        [canonical_by_case[case_id]["predicted_labels"] for case_id in frozen.case_ids],
    )
    methods = {
        "opdiag": _summary(
            rows_by_method["opdiag"],
            labels,
            {
                "display_name": "OpDiag",
                "description": (
                    "OpDiag's paper-defined independent KPI anomaly random forests "
                    "with a training-fold Top-k root-count output adapter."
                ),
                "uses_h_matrix": False,
                "uses_llm": False,
                "root_detection_matches_paper": True,
                "multi_root_extension": (
                    "training-fold root-count model followed by Top-k anomaly probabilities"
                ),
                "sql_operator_attribution_evaluated": False,
            },
        ),
        "dbaiops": _summary(
            rows_by_method["dbaiops"],
            labels,
            {
                "display_name": "DBAIOps",
                "description": (
                    "Paper-described correlation-aware anomaly models, ExperienceGraph, "
                    "two-stage graph evolution, and training-fold Top-k root output."
                ),
                "uses_h_matrix": False,
                "uses_experience_graph": True,
                "uses_llm": False,
                "reproduction_scope": (
                    "DB-MAGS closed-label diagnosis; graph-augmented root ranking replaces "
                    "the paper's free-form report generation."
                ),
            },
        ),
        canonical_method: _summary(
            rows_by_method[canonical_method],
            labels,
            {
                "display_name": specification["display_name"],
                "description": specification["description"],
                "uses_h_matrix": True,
                "uses_epdg_evidence": True,
                "uses_llm": canonical_method == "hyperdbdiag",
                "pipeline": canonical.get("stage", {}),
            },
        ),
    }
    output = {
        "protocol": {
            "name": "dbmags_leave_replicate_paper_aligned_main_comparison_v6",
            "seed": int(seed),
            "frozen_protocol": FROZEN_PROTOCOL,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "feature_names": list(feature_names),
            "outer_split": "leave-one-replicate-index-out",
            "outer_split_count": frozen.replicate_count,
            "training_sample_count_per_split": 550,
            "evaluation_sample_count_per_split": 110,
            "canonical_method": canonical_method,
            "scenario_generalization": {
                "status": "not_estimand",
                "reason": (
                    "The registered leave-one-replicate-index-out protocol retains all 22 "
                    "scenario identities in both training and evaluation partitions; it "
                    "measures repeated-block diagnosis, not unseen-scenario transfer."
                ),
                "diagnostic_leave_one_scenario_out_opdiag_exact_mean": 0.7151515151515153,
                "diagnostic_is_not_reported_as_main_result": True,
            },
            "comparability": (
                "All evaluated methods use the same 660 cases, six outer folds, and root labels. "
                "OpDiag root detection uses frozen KPIs; DBAIOps derives its anomaly models "
                "and ExperienceGraph from the same training KPIs; EPDG-grounded rows additionally "
                "use integrity-bound anonymous SQL-shape evidence, while the complete HyperDBDiag "
                "row also includes constrained LLM arbitration."
            ),
        },
        "folds": folds,
        "methods": methods,
        "paired_comparisons": {
            f"{canonical_method}_minus_opdiag": _paired_bootstrap(
                rows_by_method["opdiag"],
                rows_by_method[canonical_method],
                seed + 11,
                bootstrap_repetitions,
            ),
            f"{canonical_method}_minus_dbaiops": _paired_bootstrap(
                rows_by_method["dbaiops"],
                rows_by_method[canonical_method],
                seed + 12,
                bootstrap_repetitions,
            ),
        },
    }
    _write_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbmags-root", type=Path, default=DEFAULT_DBMAGS_ROOT)
    parser.add_argument(
        "--ablation-report", type=Path, default=DEFAULT_ABLATION_REPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument(
        "--canonical-method", choices=tuple(CANONICAL_METHODS), default="hyperdbdiag"
    )
    args = parser.parse_args()
    report = run(
        args.dbmags_root,
        args.ablation_report,
        args.output,
        args.seed,
        args.bootstrap_repetitions,
        args.canonical_method,
    )
    print(
        json.dumps(
            {
                method: {
                    "exact": values["overall"]["exact_set_accuracy"],
                    "component_f1": values["overall"]["component_f1"],
                }
                for method, values in report["methods"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
