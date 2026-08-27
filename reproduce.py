#!/usr/bin/env python3
"""Run and verify every paper-facing HyperDBDiag experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DEFAULT_OUTPUT = ROOT / "reproduced"


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith("HYPERDBDIAG_LLM_") or name in {
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_MODEL",
            "OPENAI_REASONING_EFFORT",
        }:
            env.pop(name, None)
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(SRC),
        }
    )
    return env


def _run(arguments: Sequence[str], env: Mapping[str, str]) -> None:
    command = [sys.executable, *arguments]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=dict(env), check=True)


def _load(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{name}: expected {expected}, received {actual}")


def _verify(output: Path) -> Mapping[str, Any]:
    ablation = _load(output / "ablation.json")
    ablation_methods = next(iter(ablation["datasets"].values()))["methods"]
    expected_ablation = {
        "ordinary_binary_graph": (0.42424242424242425, 0.6965394853593612),
        "hypergraph_without_epdg": (0.8560606060606061, 0.9100378787878788),
        "hypergraph": (0.9060606060606060, 0.9412878787878788),
        "hypergraph_local_judge": (0.9060606060606060, 0.9412878787878788),
    }
    for method, (exact, f1) in expected_ablation.items():
        overall = ablation_methods[method]["overall"]
        _check(f"ablation/{method}/exact", overall["exact_set_accuracy"], exact)
        _check(f"ablation/{method}/f1", overall["component_f1"], f1)
    for method in ("hypergraph_llm", "hyperdbdiag"):
        status = ablation_methods[method]["stage"]["estimand_status"]
        if status != "not_run_missing_explicit_llm_configuration":
            raise AssertionError(f"deterministic {method} unexpectedly used an LLM")

    main = _load(output / "main_registered_llm.json")["methods"]
    expected_main = {
        "opdiag": (0.8484848484848485, 0.9053030303030303),
        "dbaiops": (0.8500000000000000, 0.9067676289635589),
        "hyperdbdiag": (0.9196969696969697, 0.9498106060606061),
    }
    for method, (exact, f1) in expected_main.items():
        overall = main[method]["overall"]
        _check(f"main/{method}/exact", overall["exact_set_accuracy"], exact)
        _check(f"main/{method}/f1", overall["component_f1"], f1)

    operator = _load(output / "operator.json")["methods"]
    opdiag = operator["opdiag_dbmags_reproduction"]["metrics"]
    hyperdbdiag = operator["hyperdbdiag"]["metrics"]
    _check("operator/opdiag/sql", opdiag["root_sql_pairs"]["hit_at_1"], 0.5086805555555556)
    _check("operator/opdiag/operator", opdiag["root_operator_pairs"]["hit_at_1"], 0.75)
    _check("operator/hyperdbdiag/sql", hyperdbdiag["root_sql_pairs"]["hit_at_1"], 0.9375)
    _check(
        "operator/hyperdbdiag/operator",
        hyperdbdiag["root_operator_pairs"]["hit_at_1"],
        0.9166666666666666,
    )

    structure = _load(output / "structure.json")["aggregate"]
    _check("structure/hypergraph_relations", structure["mean_hypergraph_incidence_count"], 2879.1666666666665)
    _check("structure/pairwise_relations", structure["mean_equivalent_pairwise_occurrence_count"], 7584.0)
    _check("structure/relation_reduction", structure["relation_item_reduction"], 0.6203630450070323)

    return {
        "status": "verified",
        "main_exact": {
            method: values["overall"]["exact_set_accuracy"]
            for method, values in main.items()
        },
        "fine_grained_hit_at_1": {
            "opdiag_sql": opdiag["root_sql_pairs"]["hit_at_1"],
            "opdiag_operator": opdiag["root_operator_pairs"]["hit_at_1"],
            "hyperdbdiag_sql": hyperdbdiag["root_sql_pairs"]["hit_at_1"],
            "hyperdbdiag_operator": hyperdbdiag["root_operator_pairs"]["hit_at_1"],
        },
        "structure": structure,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = _environment()

    if not args.skip_tests:
        _run(["-m", "unittest", "discover", "-s", "tests", "-v"], env)
    _run(
        [
            "src/hyperdbdiag_ablation.py",
            "--seed",
            "20260802",
            "--output",
            str(output / "ablation.json"),
        ],
        env,
    )
    _run(
        [
            "src/dbmags_experiment.py",
            "--seed",
            "20260802",
            "--canonical-method",
            "epdg_hypergraph",
            "--ablation-report",
            str(output / "ablation.json"),
            "--output",
            str(output / "main_deterministic.json"),
        ],
        env,
    )
    _run(
        [
            "src/dbmags_experiment.py",
            "--seed",
            "20260802",
            "--canonical-method",
            "hyperdbdiag",
            "--ablation-report",
            "runs/dbmags-ablation/full_report_llm_xhigh.json",
            "--output",
            str(output / "main_registered_llm.json"),
        ],
        env,
    )
    _run(
        [
            "src/operator_bound_experiment.py",
            "--dataset-root",
            "data/dbmags_operator_bound_v4",
            "--audit-only",
        ],
        env,
    )
    _run(
        [
            "src/operator_bound_experiment.py",
            "--dataset-root",
            "data/dbmags_operator_bound_v4",
            "--output",
            str(output / "operator.json"),
        ],
        env,
    )
    _run(
        [
            "src/structure_efficiency_experiment.py",
            "--output",
            str(output / "structure.json"),
        ],
        env,
    )
    print(json.dumps(_verify(output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
