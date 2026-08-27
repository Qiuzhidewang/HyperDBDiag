"""Read frozen, label-free SQL-shape observations for DB-MAGS cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple


FROZEN_SEMANTIC_PROTOCOL = "dbmags-frozen-anonymous-semantic-evidence-v1"


@dataclass(frozen=True)
class SemanticObservation:
    """One anonymous query-shape observation from a runtime source."""

    atoms: Tuple[str, ...]
    source_channels: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.atoms or len(set(self.atoms)) != len(self.atoms):
            raise ValueError("semantic observations require unique atoms")
        if not self.source_channels or len(set(self.source_channels)) != len(
            self.source_channels
        ):
            raise ValueError("semantic observations require a source channel")

    def summary(self) -> str:
        return (
            "Anonymous runtime SQL-shape observation. Operators and clauses: "
            + ", ".join(self.atoms)
            + ". Source channels: "
            + ", ".join(self.source_channels)
            + ". No SQL text, identifiers, counts, or labels are included."
        )


def load_frozen_case_observations(
    path: Path, case_ids: Sequence[str]
) -> Dict[str, Tuple[SemanticObservation, ...]]:
    """Load the integrity-bound anonymous observations for every frozen case."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != {"protocol", "samples"}:
        raise ValueError("frozen semantic evidence contains an unregistered field")
    if payload.get("protocol") != FROZEN_SEMANTIC_PROTOCOL:
        raise ValueError("unexpected frozen semantic evidence protocol")
    result: Dict[str, Tuple[SemanticObservation, ...]] = {}
    for row in payload.get("samples") or ():
        if not isinstance(row, Mapping) or set(row) != {"case_id", "observations"}:
            raise ValueError("frozen semantic sample is malformed")
        case_id = str(row["case_id"])
        if case_id in result:
            raise ValueError("frozen semantic case IDs are not unique")
        observations = []
        for observation in row["observations"]:
            if not isinstance(observation, Mapping) or set(observation) != {
                "atoms",
                "source_channels",
            }:
                raise ValueError("frozen semantic observation is malformed")
            observations.append(
                SemanticObservation(
                    tuple(str(value) for value in observation["atoms"]),
                    tuple(str(value) for value in observation["source_channels"]),
                )
            )
        result[case_id] = tuple(observations)
    if set(result) != {str(case_id) for case_id in case_ids}:
        raise ValueError("frozen metric and semantic case inventories differ")
    return result


def semantic_inventory(
    observations: Iterable[SemanticObservation],
) -> Tuple[str, ...]:
    """Return the sorted atom inventory for an audit report."""

    return tuple(sorted({atom for row in observations for atom in row.atoms}))
