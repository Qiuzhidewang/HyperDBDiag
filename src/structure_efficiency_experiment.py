"""Measure DB-MAGS hypergraph size and isolated structure-traversal cost."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np

from hypergraph_core import HypergraphDiffusionResidualEncoder
from metric_frozen_dataset import load_frozen_metric_dataset


DEFAULT_DBMAGS_ROOT = Path("data/dbmags_interaction_v10_metric_only")
DEFAULT_OUTPUT = Path("runs/dbmags-structure-efficiency/full_report.json")
DEFAULT_WARMUP_ROUNDS = 10
DEFAULT_MEASURED_REPETITIONS = 15
DEFAULT_BATCH_ITERATIONS = 20


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _relation_indices(
    active_incidence: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = np.asarray(active_incidence, dtype=bool)
    if active.ndim != 2 or not active.shape[0] or not active.shape[1]:
        raise ValueError("active incidence must be a nonempty two-dimensional matrix")

    vertex_ids = []
    hyperedge_ids = []
    pair_left = []
    pair_right = []
    for hyperedge_id in range(active.shape[1]):
        vertices = np.flatnonzero(active[:, hyperedge_id]).astype(np.intp)
        if not len(vertices):
            raise ValueError("every training hyperedge must contain an active vertex")
        vertex_ids.extend(vertices.tolist())
        hyperedge_ids.extend([hyperedge_id] * len(vertices))
        left, right = np.triu_indices(len(vertices), k=1)
        pair_left.extend(vertices[left].tolist())
        pair_right.extend(vertices[right].tolist())
    return tuple(
        np.asarray(values, dtype=np.intp)
        for values in (vertex_ids, hyperedge_ids, pair_left, pair_right)
    )


def _hypergraph_traversal(
    query_signals: np.ndarray,
    vertex_ids: np.ndarray,
    hyperedge_ids: np.ndarray,
    hyperedge_degrees: np.ndarray,
    vertex_count: int,
    hyperedge_count: int,
) -> float:
    checksum = 0.0
    for signal in query_signals:
        edge_values = np.bincount(
            hyperedge_ids,
            weights=signal[vertex_ids],
            minlength=hyperedge_count,
        ) / hyperedge_degrees
        checksum += float(
            np.sum(
                np.bincount(
                    vertex_ids,
                    weights=edge_values[hyperedge_ids],
                    minlength=vertex_count,
                )
            )
        )
    return checksum


def _pairwise_traversal(
    query_signals: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    vertex_count: int,
) -> float:
    checksum = 0.0
    for signal in query_signals:
        checksum += float(
            np.sum(
                np.bincount(
                    pair_left,
                    weights=signal[pair_right],
                    minlength=vertex_count,
                )
                + np.bincount(
                    pair_right,
                    weights=signal[pair_left],
                    minlength=vertex_count,
                )
            )
        )
    return checksum


def _measure_ms(function: Callable[[], float], batch_iterations: int) -> float:
    started = perf_counter_ns()
    checksum = 0.0
    for _ in range(batch_iterations):
        checksum += function()
    elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0 / batch_iterations
    if not np.isfinite(checksum):
        raise RuntimeError("structure traversal produced a non-finite checksum")
    return float(elapsed_ms)


def _benchmark_pair(
    hypergraph: Callable[[], float],
    pairwise: Callable[[], float],
    warmup_rounds: int,
    measured_repetitions: int,
    batch_iterations: int,
) -> Dict[str, Any]:
    for _ in range(warmup_rounds):
        hypergraph()
        pairwise()

    hypergraph_ms = []
    pairwise_ms = []
    for repetition in range(measured_repetitions):
        if repetition % 2:
            pairwise_ms.append(_measure_ms(pairwise, batch_iterations))
            hypergraph_ms.append(_measure_ms(hypergraph, batch_iterations))
        else:
            hypergraph_ms.append(_measure_ms(hypergraph, batch_iterations))
            pairwise_ms.append(_measure_ms(pairwise, batch_iterations))
    hypergraph_median = float(np.median(hypergraph_ms))
    pairwise_median = float(np.median(pairwise_ms))
    return {
        "hypergraph_median_ms": hypergraph_median,
        "pairwise_median_ms": pairwise_median,
        "absolute_saving_ms": pairwise_median - hypergraph_median,
        "relative_time_reduction": 1.0 - hypergraph_median / pairwise_median,
        "hypergraph_measurements_ms": hypergraph_ms,
        "pairwise_measurements_ms": pairwise_ms,
    }


def run(
    dbmags_root: Path = DEFAULT_DBMAGS_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
    measured_repetitions: int = DEFAULT_MEASURED_REPETITIONS,
    batch_iterations: int = DEFAULT_BATCH_ITERATIONS,
) -> Dict[str, Any]:
    if min(warmup_rounds, measured_repetitions, batch_iterations) < 1:
        raise ValueError("all timing controls must be positive")
    frozen = load_frozen_metric_dataset(dbmags_root)
    folds = []
    for held_out_replicate in range(1, frozen.replicate_count + 1):
        train_ids = tuple(
            case_id
            for case_id in frozen.case_ids
            if frozen.replicate_by_case[case_id] != held_out_replicate
        )
        evaluation_ids = tuple(
            case_id
            for case_id in frozen.case_ids
            if frozen.replicate_by_case[case_id] == held_out_replicate
        )
        train_x = np.asarray([frozen.features[case_id] for case_id in train_ids])
        evaluation_x = np.asarray([frozen.features[case_id] for case_id in evaluation_ids])
        encoder = HypergraphDiffusionResidualEncoder().fit(train_x)
        if encoder.scaler is None:
            raise RuntimeError("the training-fold scaler was not fitted")
        training_incidence = encoder._incidence(encoder.scaler.transform(train_x))
        query_signals = encoder._incidence(encoder.scaler.transform(evaluation_x)).T
        active_incidence = np.abs(training_incidence) > 0.0
        vertex_ids, hyperedge_ids, pair_left, pair_right = _relation_indices(
            active_incidence
        )
        vertex_count, hyperedge_count = active_incidence.shape
        hyperedge_degrees = np.bincount(
            hyperedge_ids, minlength=hyperedge_count
        ).astype(np.float64)
        hypergraph = lambda: _hypergraph_traversal(
            query_signals,
            vertex_ids,
            hyperedge_ids,
            hyperedge_degrees,
            vertex_count,
            hyperedge_count,
        )
        pairwise = lambda: _pairwise_traversal(
            query_signals,
            pair_left,
            pair_right,
            vertex_count,
        )
        timing = _benchmark_pair(
            hypergraph,
            pairwise,
            warmup_rounds,
            measured_repetitions,
            batch_iterations,
        )
        folds.append(
            {
                "outer_split": f"leave_replicate_index_out:{held_out_replicate}",
                "training_case_count": len(train_ids),
                "evaluation_case_count": len(evaluation_ids),
                "signed_metric_vertex_count": int(vertex_count),
                "training_hyperedge_count": int(hyperedge_count),
                "hypergraph_incidence_count": int(len(vertex_ids)),
                "equivalent_pairwise_occurrence_count": int(len(pair_left)),
                "relation_item_reduction": 1.0 - len(vertex_ids) / float(len(pair_left)),
                "timing": timing,
            }
        )

    incidence_total = sum(row["hypergraph_incidence_count"] for row in folds)
    pairwise_total = sum(row["equivalent_pairwise_occurrence_count"] for row in folds)
    hypergraph_time = float(
        np.mean([row["timing"]["hypergraph_median_ms"] for row in folds])
    )
    pairwise_time = float(
        np.mean([row["timing"]["pairwise_median_ms"] for row in folds])
    )
    report = {
        "protocol": {
            "name": "dbmags_equivalent_pairwise_expansion_vs_metric_hypergraph",
            "dataset": str(dbmags_root),
            "outer_split": "six-fold leave-one-replicate-index-out",
            "comparison_scope": "isolated relationship representation and one structure traversal over each evaluation fold",
            "hypergraph_relation_item": "one nonzero signed-metric-atom-to-training-hyperedge incidence",
            "pairwise_relation_item": "one undirected metric-atom pair occurrence for every pair inside each training hyperedge; repeated pairs remain separate to preserve hyperedge-specific co-occurrence",
            "excluded_from_timing": [
                "data loading",
                "standardization and incidence construction",
                "classifier and candidate decoding",
                "EPDG",
                "local structured judge",
                "LLM arbitration",
            ],
            "timing": {
                "warmup_rounds": warmup_rounds,
                "measured_repetitions": measured_repetitions,
                "batch_iterations_per_measurement": batch_iterations,
                "summary": "median within each fold, then arithmetic mean across folds",
                "alternating_measurement_order": True,
            },
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
        },
        "aggregate": {
            "fold_count": len(folds),
            "mean_hypergraph_incidence_count": incidence_total / float(len(folds)),
            "mean_equivalent_pairwise_occurrence_count": pairwise_total / float(len(folds)),
            "relation_item_reduction": 1.0 - incidence_total / float(pairwise_total),
            "mean_hypergraph_traversal_ms": hypergraph_time,
            "mean_pairwise_traversal_ms": pairwise_time,
            "absolute_traversal_saving_ms": pairwise_time - hypergraph_time,
            "relative_traversal_time_reduction": 1.0 - hypergraph_time / pairwise_time,
        },
        "folds": folds,
    }
    _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbmags-root", type=Path, default=DEFAULT_DBMAGS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup-rounds", type=int, default=DEFAULT_WARMUP_ROUNDS)
    parser.add_argument(
        "--measured-repetitions", type=int, default=DEFAULT_MEASURED_REPETITIONS
    )
    parser.add_argument("--batch-iterations", type=int, default=DEFAULT_BATCH_ITERATIONS)
    args = parser.parse_args()
    report = run(
        dbmags_root=args.dbmags_root,
        output_path=args.output,
        warmup_rounds=args.warmup_rounds,
        measured_repetitions=args.measured_repetitions,
        batch_iterations=args.batch_iterations,
    )
    print(json.dumps(report["aggregate"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
