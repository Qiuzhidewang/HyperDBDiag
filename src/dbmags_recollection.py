"""Collect a traceable DB-MAGS interaction cohort from a live MySQL system.

The collector keeps truth labels and predictor evidence on opposite sides of
the collection boundary. Root labels schedule the injected mechanisms, while
the frozen semantic observations are derived only from statements, plans, and
runtime lock observations captured during the case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pymysql


PROTOCOL = "dbmags-interaction-v11-window-complete"
DEFAULT_OUTPUT = Path("data/dbmags_interaction_v11_recollected")
TABLES = ("customer", "orders")
PAIR_ROOTS = (
    ("missing_index", "implicit_conversion"),
    ("missing_index", "query_with_too_much_join"),
    ("missing_index", "order_by"),
    ("missing_index", "group_by"),
    ("missing_index", "query_whole_table"),
    ("record_lock", "implicit_conversion"),
    ("record_lock", "query_with_too_much_join"),
    ("record_lock", "order_by"),
    ("record_lock", "group_by"),
    ("record_lock", "query_whole_table"),
    ("record_lock", "missing_index"),
)
CONDITIONS = ("single_a", "single_b", "a_then_b", "b_then_a", "overlap")
REQUIRED_EVIDENCE_ATOM = {
    "missing_index": "broad_filtered_scan",
    "implicit_conversion": "predicate_type_conversion",
    "query_with_too_much_join": "multi_table_join",
    "order_by": "ordering_clause",
    "group_by": "aggregation_clause",
    "query_whole_table": "unrestricted_relation_read",
    "record_lock": "row_lock_wait",
}
METRICS = (
    "mysql_data_locks",
    "mysql_data_lock_waits",
    "mysql_innodb_row_lock_current_waits",
    "mysql_threads_running",
    "mysql_active_sessions",
)
TIME_BINS_SECONDS = ((0.0, 3.0), (3.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 75.0))
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING = re.compile(r"'(?:''|[^'])*'")
_NUMBER = re.compile(r"(?<![A-Za-z_])\b\d+(?:\.\d+)?\b")
_SPACE = re.compile(r"\s+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _normalise_sql(sql: str) -> str:
    text = _COMMENT.sub(" ", sql)
    text = _STRING.sub("?", text)
    text = _NUMBER.sub("?", text)
    return _SPACE.sub(" ", text).strip().lower()


def _walk_plan(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_plan(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_plan(child)


def _plan_tables(plan: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [node for node in _walk_plan(plan) if "table_name" in node and "access_type" in node]


def _plan_flag(plan: Mapping[str, Any], key: str) -> bool:
    return any(bool(node.get(key)) for node in _walk_plan(plan))


def _semantic_atoms(sql: str, plan: Mapping[str, Any]) -> Tuple[str, ...]:
    """Extract anonymous SQL/plan atoms without reading labels or case metadata."""

    lowered = _COMMENT.sub(" ", sql).lower()
    tables = _plan_tables(plan)
    atoms = set()
    has_where = bool(re.search(r"\bwhere\b", lowered))
    if has_where:
        atoms.add("filtered_relation_read")
        if re.search(r"(?:=|<>|!=|<=|>=|<|>|\bbetween\b|\blike\b|\bin\s*\()", lowered):
            atoms.add("range_or_comparison_predicate")
    if re.search(r"\bgroup\s+by\b", lowered) and any(
        "grouping_operation" in node for node in _walk_plan(plan)
    ):
        atoms.add("aggregation_clause")
    if re.search(r"\border\s+by\b", lowered) and _plan_flag(plan, "using_filesort"):
        atoms.add("ordering_clause")
    if len(tables) >= 4:
        atoms.add("multi_table_join")
    if not has_where and any(str(row.get("access_type", "")).upper() == "ALL" for row in tables):
        atoms.add("unrestricted_relation_read")
    if has_where and any(
        str(row.get("access_type", "")).upper() not in {"CONST", "EQ_REF", "SYSTEM"}
        and int(row.get("rows_examined_per_scan") or 0) >= 1000
        for row in tables
    ):
        atoms.add("broad_filtered_scan")
    if re.search(r"\bc_last\s*=\s*\d+\b|\bo_entry_d\s*=\s*\d{8,}\b", lowered):
        atoms.add("predicate_type_conversion")
    if re.search(r"\b(?:update|delete\s+from|select)\b", lowered) and (
        " for update" in lowered or lowered.lstrip().startswith(("update ", "delete "))
    ):
        atoms.add("row_locking_clause")
    return tuple(sorted(atoms))


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    def connect(self, *, autocommit: bool = True) -> pymysql.Connection:
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            connect_timeout=10,
            read_timeout=15,
            write_timeout=15,
            autocommit=autocommit,
        )


def _container_password(container: str) -> str:
    payload = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
    environment = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in payload["Config"].get("Env", ())
        if "=" in item
    }
    # The preserved local DB-MAGS container was created with socket/host root
    # access and no stored password. New deployments can provide the password
    # through DBMAGS_MYSQL_PASSWORD or MYSQL_ROOT_PASSWORD without persisting it.
    return environment.get("MYSQL_ROOT_PASSWORD", "")


@dataclass
class StatementStats:
    sql: str
    count: int = 0
    total_seconds: float = 0.0
    rows_seen: int = 0
    errors: int = 0

    def add(self, elapsed: float, rows: int, error: bool) -> None:
        self.count += 1
        self.total_seconds += max(float(elapsed), 0.0)
        self.rows_seen += max(int(rows), 0)
        self.errors += int(error)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sql": self.sql,
            "count": self.count,
            "total_seconds": self.total_seconds,
            "rows_seen": self.rows_seen,
            "errors": self.errors,
        }


class StatementTrace:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = "baseline"
        self._rows: MutableMapping[str, MutableMapping[str, StatementStats]] = defaultdict(dict)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)

    def add(self, sql: str, elapsed: float, rows: int, error: bool = False) -> None:
        signature = _normalise_sql(sql)
        with self._lock:
            stats = self._rows[self._phase].get(signature)
            if stats is None:
                stats = StatementStats(sql=sql)
                self._rows[self._phase][signature] = stats
            stats.add(elapsed, rows, error)

    def snapshot(self, phase: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                key: value.as_dict()
                for key, value in sorted(self._rows.get(phase, {}).items())
            }


class Workload(threading.Thread):
    def __init__(self, config: MySQLConfig, trace: StatementTrace, seed: int) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.trace = trace
        self.random = random.Random(seed)
        self.stop_event = threading.Event()
        self.failure: Optional[str] = None

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        statements = (
            "SELECT c_balance, c_discount FROM customer WHERE c_w_id=%s AND c_d_id=%s AND c_id=%s",
            "SELECT o_id, o_entry_d FROM orders WHERE o_w_id=%s AND o_d_id=%s AND o_c_id=%s ORDER BY o_id DESC LIMIT 1",
            "SELECT COUNT(*) FROM order_line WHERE ol_w_id=%s AND ol_d_id=%s AND ol_o_id BETWEEN %s AND %s",
            "SELECT s_quantity FROM stock WHERE s_w_id=%s AND s_i_id=%s",
        )
        try:
            connection = self.config.connect()
            with connection:
                while not self.stop_event.is_set():
                    index = self.random.randrange(len(statements))
                    if index == 0:
                        params = (self.random.randint(1, 4), self.random.randint(1, 10), self.random.randint(1, 3000))
                    elif index == 1:
                        params = (self.random.randint(1, 4), self.random.randint(1, 10), self.random.randint(1, 3000))
                    elif index == 2:
                        start = self.random.randint(1, 2990)
                        params = (self.random.randint(1, 4), self.random.randint(1, 10), start, start + 10)
                    else:
                        params = (self.random.randint(1, 4), self.random.randint(1, 100000))
                    started = time.monotonic()
                    error = False
                    rows = 0
                    rendered: Any = statements[index]
                    try:
                        with connection.cursor() as cursor:
                            rendered = cursor.mogrify(statements[index], params)
                            cursor.execute(statements[index], params)
                            rows = len(cursor.fetchall())
                    except Exception:
                        error = True
                        connection.ping(reconnect=True)
                    rendered_sql = rendered.decode() if isinstance(rendered, bytes) else str(rendered)
                    self.trace.add(rendered_sql, time.monotonic() - started, rows, error)
                    self.stop_event.wait(0.05)
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"


def _query_for(root: str, table: str, replicate: int) -> str:
    warehouse = (replicate - 1) % 4 + 1
    district = (replicate * 3 - 1) % 10 + 1
    if root == "missing_index":
        return (
            f"SELECT COUNT(*) FROM customer WHERE c_state='ZZ' AND c_w_id={warehouse}"
            if table == "customer"
            else f"SELECT COUNT(*) FROM orders WHERE o_carrier_id={replicate + 20}"
        )
    if root == "implicit_conversion":
        return (
            f"SELECT COUNT(*) FROM customer WHERE c_w_id={warehouse} AND c_d_id={district} AND c_last={90000 + replicate}"
            if table == "customer"
            else f"SELECT COUNT(*) FROM orders WHERE o_entry_d={20250101000000 + replicate}"
        )
    if root == "order_by":
        return (
            f"SELECT c_id, c_balance FROM customer WHERE c_w_id={warehouse} ORDER BY c_balance LIMIT 2000"
            if table == "customer"
            else f"SELECT o_id, o_entry_d FROM orders WHERE o_w_id={warehouse} ORDER BY o_entry_d LIMIT 2000"
        )
    if root == "group_by":
        return (
            "SELECT c_state, COUNT(*), SUM(c_balance) FROM customer GROUP BY c_state"
            if table == "customer"
            else "SELECT o_carrier_id, COUNT(*), SUM(o_ol_cnt) FROM orders GROUP BY o_carrier_id"
        )
    if root == "query_whole_table":
        return (
            "SELECT SUM(c_balance) FROM customer"
            if table == "customer"
            else "SELECT SUM(o_ol_cnt) FROM orders"
        )
    if root == "query_with_too_much_join":
        return (
            "SELECT COUNT(*) FROM customer c JOIN orders o ON o.o_w_id=c.c_w_id AND o.o_d_id=c.c_d_id AND o.o_c_id=c.c_id "
            "JOIN order_line ol ON ol.ol_w_id=o.o_w_id AND ol.ol_d_id=o.o_d_id AND ol.ol_o_id=o.o_id "
            "JOIN stock s ON s.s_w_id=ol.ol_supply_w_id AND s.s_i_id=ol.ol_i_id "
            f"JOIN item i ON i.i_id=s.s_i_id WHERE c.c_w_id={warehouse} AND c.c_d_id={district} AND c.c_id BETWEEN 1 AND 120"
        )
    raise ValueError(f"root has no query mechanism: {root}")


def _lock_sql(table: str, replicate: int) -> Tuple[str, Tuple[int, ...]]:
    warehouse = (replicate - 1) % 4 + 1
    district = (replicate * 3 - 1) % 10 + 1
    row_id = (replicate * 397) % 2900 + 1
    if table == "customer":
        return (
            "UPDATE customer SET c_balance=c_balance WHERE c_w_id=%s AND c_d_id=%s AND c_id=%s",
            (warehouse, district, row_id),
        )
    return (
        "UPDATE orders SET o_carrier_id=o_carrier_id WHERE o_w_id=%s AND o_d_id=%s AND o_id=%s",
        (warehouse, district, row_id),
    )


class RootRunner(threading.Thread):
    def __init__(self, config: MySQLConfig, trace: StatementTrace, root: str, table: str, replicate: int, duration: float) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.trace = trace
        self.root = root
        self.table = table
        self.replicate = replicate
        self.duration = duration
        self.executions = 0
        self.errors: List[str] = []
        self.connection_ids: List[int] = []
        self.lock_wait_observed = False

    def _query_loop(self) -> None:
        sql = _query_for(self.root, self.table, self.replicate)
        deadline = time.monotonic() + self.duration
        connection = self.config.connect()
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SET SESSION MAX_EXECUTION_TIME=10000")
                cursor.execute("SELECT CONNECTION_ID()")
                self.connection_ids.append(int(cursor.fetchone()[0]))
            while time.monotonic() < deadline:
                started = time.monotonic()
                error = False
                rows = 0
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(sql)
                        rows = len(cursor.fetchall())
                except Exception as exc:
                    error = True
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                    connection.ping(reconnect=True)
                self.trace.add(sql, time.monotonic() - started, rows, error)
                self.executions += 1
                time.sleep(0.05)

    def _lock_loop(self) -> None:
        sql, params = _lock_sql(self.table, self.replicate)
        deadline = time.monotonic() + self.duration
        while time.monotonic() < deadline:
            holder = self.config.connect(autocommit=False)
            waiter = self.config.connect(autocommit=False)
            waiter_result: Dict[str, Any] = {}
            try:
                with holder.cursor() as cursor:
                    cursor.execute("SELECT CONNECTION_ID()")
                    self.connection_ids.append(int(cursor.fetchone()[0]))
                    started = time.monotonic()
                    rendered = cursor.mogrify(sql, params)
                    cursor.execute(sql, params)
                    rendered_sql = rendered.decode() if isinstance(rendered, bytes) else str(rendered)
                    self.trace.add(rendered_sql, time.monotonic() - started, cursor.rowcount)
                with waiter.cursor() as cursor:
                    cursor.execute("SELECT CONNECTION_ID()")
                    self.connection_ids.append(int(cursor.fetchone()[0]))
                    cursor.execute("SET SESSION innodb_lock_wait_timeout=4")

                def blocked_update() -> None:
                    started = time.monotonic()
                    error = False
                    rows = 0
                    rendered: Any = sql
                    try:
                        with waiter.cursor() as cursor:
                            rendered = cursor.mogrify(sql, params)
                            cursor.execute(sql, params)
                            rows = cursor.rowcount
                    except Exception as exc:
                        error = True
                        waiter_result["error"] = f"{type(exc).__name__}: {exc}"
                    waiter_result["elapsed"] = time.monotonic() - started
                    rendered_sql = rendered.decode() if isinstance(rendered, bytes) else str(rendered)
                    self.trace.add(rendered_sql, waiter_result["elapsed"], rows, error)

                thread = threading.Thread(target=blocked_update, daemon=True)
                thread.start()
                time.sleep(0.4)
                audit = self.config.connect()
                with audit:
                    with audit.cursor() as cursor:
                        cursor.execute("SELECT COUNT(*) FROM performance_schema.data_lock_waits")
                        self.lock_wait_observed |= int(cursor.fetchone()[0]) > 0
                time.sleep(0.8)
                holder.rollback()
                thread.join(timeout=6)
                waiter.rollback()
                self.executions += 1
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                try:
                    holder.rollback()
                    waiter.rollback()
                except Exception:
                    pass
            finally:
                holder.close()
                waiter.close()

    def run(self) -> None:
        try:
            if self.root == "record_lock":
                self._lock_loop()
            else:
                self._query_loop()
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "execution_count": self.executions,
            "connection_ids": sorted(set(self.connection_ids)),
            "lock_wait_observed": self.lock_wait_observed,
            "errors": self.errors,
        }


def _metric_sample(connection: pymysql.Connection) -> Dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM performance_schema.data_locks")
        data_locks = int(cursor.fetchone()[0])
        cursor.execute("SELECT COUNT(*) FROM performance_schema.data_lock_waits")
        data_lock_waits = int(cursor.fetchone()[0])
        cursor.execute("SELECT COUNT FROM information_schema.innodb_metrics WHERE NAME='lock_row_lock_current_waits'")
        current_waits_row = cursor.fetchone()
        cursor.execute("SHOW GLOBAL STATUS LIKE 'Threads_running'")
        threads_running = int(cursor.fetchone()[1])
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.processlist "
            "WHERE COMMAND <> 'Sleep' AND ID <> CONNECTION_ID()"
        )
        active_sessions = int(cursor.fetchone()[0])
    return {
        "mysql_data_locks": data_locks,
        "mysql_data_lock_waits": data_lock_waits,
        "mysql_innodb_row_lock_current_waits": int(current_waits_row[0]) if current_waits_row else 0,
        "mysql_threads_running": threads_running,
        "mysql_active_sessions": active_sessions,
    }


def _sample_for(config: MySQLConfig, duration: float, reference: float, interval: float = 1.0) -> List[Dict[str, Any]]:
    rows = []
    connection = config.connect()
    with connection:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            started = time.monotonic()
            row = _metric_sample(connection)
            row["relative_seconds"] = time.monotonic() - reference
            rows.append(row)
            time.sleep(max(0.0, interval - (time.monotonic() - started)))
    return rows


def _explain(config: MySQLConfig, sql: str) -> Mapping[str, Any]:
    connection = config.connect()
    with connection:
        with connection.cursor() as cursor:
            cursor.execute("EXPLAIN FORMAT=JSON " + sql)
            value = cursor.fetchone()[0]
    return json.loads(value)


def _schedule(condition: str, root_a: str, root_b: str, root_duration: float) -> Tuple[Tuple[str, float], ...]:
    if condition == "single_a":
        return ((root_a, 0.0),)
    if condition == "single_b":
        return ((root_b, 0.0),)
    if condition == "a_then_b":
        return ((root_a, 0.0), (root_b, root_duration))
    if condition == "b_then_a":
        return ((root_b, 0.0), (root_a, root_duration))
    if condition == "overlap":
        return ((root_a, 0.0), (root_b, root_duration / 2.0))
    raise ValueError(f"unknown condition: {condition}")


def _select_anomalous_statements(
    baseline: Mapping[str, Mapping[str, Any]],
    anomaly: Mapping[str, Mapping[str, Any]],
    baseline_seconds: float,
    anomaly_seconds: float,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """Select anomalous statements using phase deltas only, without labels."""

    selected = []
    for signature, row in anomaly.items():
        before = baseline.get(signature, {})
        anomaly_time_rate = float(row["total_seconds"]) / max(anomaly_seconds, 1e-9)
        baseline_time_rate = float(before.get("total_seconds", 0.0)) / max(baseline_seconds, 1e-9)
        anomaly_rows_rate = float(row["rows_seen"]) / max(anomaly_seconds, 1e-9)
        baseline_rows_rate = float(before.get("rows_seen", 0.0)) / max(baseline_seconds, 1e-9)
        anomaly_count_rate = float(row["count"]) / max(anomaly_seconds, 1e-9)
        baseline_count_rate = float(before.get("count", 0.0)) / max(baseline_seconds, 1e-9)
        time_excess = max(anomaly_time_rate - baseline_time_rate, 0.0)
        rows_excess = max(anomaly_rows_rate - baseline_rows_rate, 0.0)
        count_excess = max(anomaly_count_rate - baseline_count_rate, 0.0)
        score = math.log1p(1000.0 * time_excess) + 0.15 * math.log1p(rows_excess) + 0.10 * math.log1p(count_excess)
        if score > 0.0 and int(row["count"]) > int(row.get("errors", 0)):
            selected.append({"signature": signature, "score": score, **dict(row)})
    return sorted(selected, key=lambda row: (-float(row["score"]), str(row["signature"])))[:limit]


def _expected_labels(condition: str, root_a: str, root_b: str) -> List[str]:
    if condition == "single_a":
        return [root_a]
    if condition == "single_b":
        return [root_b]
    return sorted({root_a, root_b})


def _recovery_audit(config: MySQLConfig, connection_ids: Sequence[int]) -> Dict[str, Any]:
    time.sleep(1.0)
    connection = config.connect()
    with connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM performance_schema.data_lock_waits")
            waits = int(cursor.fetchone()[0])
            if connection_ids:
                placeholders = ",".join("%s" for _ in connection_ids)
                cursor.execute(
                    f"SELECT COUNT(*) FROM information_schema.processlist WHERE ID IN ({placeholders})",
                    tuple(connection_ids),
                )
                sessions = int(cursor.fetchone()[0])
            else:
                sessions = 0
    return {"data_lock_wait_count": waits, "injector_session_count": sessions, "passed": waits == 0 and sessions == 0}


def _case_id(table: str, root_a: str, root_b: str, replicate: int, condition: str) -> str:
    return f"{table}_{root_a}_{root_b}__block_{replicate:02d}__{condition}"


def _collect_case(
    config: MySQLConfig,
    output: Path,
    table: str,
    root_a: str,
    root_b: str,
    replicate: int,
    condition: str,
    baseline_metrics: Sequence[Mapping[str, Any]],
    baseline_trace: Mapping[str, Mapping[str, Any]],
    baseline_workload_errors: Sequence[str],
    baseline_seconds: float,
    observation_seconds: float,
    root_duration: float,
    recovery_seconds: float,
    workload_count: int,
    seed: int,
) -> Dict[str, Any]:
    case_id = _case_id(table, root_a, root_b, replicate, condition)
    case_root = output / "cases" / case_id
    trace = StatementTrace()
    trace.set_phase("anomaly")
    workers = [Workload(config, trace, seed + index) for index in range(workload_count)]
    for worker in workers:
        worker.start()
    reference = time.monotonic()
    runners: List[RootRunner] = []
    schedule = _schedule(condition, root_a, root_b, root_duration)

    def launch(root: str, offset: float) -> None:
        delay = max(0.0, reference + offset - time.monotonic())
        if delay:
            time.sleep(delay)
        runner = RootRunner(config, trace, root, table, replicate, root_duration)
        runners.append(runner)
        runner.start()

    launchers = [threading.Thread(target=launch, args=row, daemon=True) for row in schedule]
    for launcher in launchers:
        launcher.start()
    anomaly_metrics = _sample_for(config, observation_seconds, reference)
    for launcher in launchers:
        launcher.join()
    for runner in runners:
        runner.join(timeout=root_duration + 15.0)
    for worker in workers:
        worker.stop()
    for worker in workers:
        worker.join(timeout=10)

    anomaly_trace = trace.snapshot("anomaly")
    selected = _select_anomalous_statements(
        baseline_trace, anomaly_trace, baseline_seconds, observation_seconds
    )
    plans = {}
    for row in selected:
        sql = str(row["sql"])
        if not sql.lstrip().lower().startswith("select"):
            continue
        try:
            plans[str(row["signature"])] = {"sql": sql, "plan": _explain(config, sql)}
        except Exception as exc:
            plans[str(row["signature"])] = {
                "sql": sql,
                "error": f"{type(exc).__name__}: {exc}",
            }
    observations = []
    for row in selected:
        plan_record = plans.get(str(row["signature"]))
        if plan_record is None or "plan" not in plan_record:
            continue
        atoms = _semantic_atoms(str(row["sql"]), plan_record["plan"])
        if atoms:
            observations.append(
                {
                    "atoms": list(atoms),
                    "source_channels": ["instrumented_statement_history", "execution_plan"],
                    "statement_signature_sha256": hashlib.sha256(str(row["signature"]).encode()).hexdigest(),
                    "anomaly_score": float(row["score"]),
                }
            )
    if any(runner.lock_wait_observed for runner in runners):
        observations.append(
            {
                "atoms": ["row_lock_wait"],
                "source_channels": ["performance_schema.data_lock_waits"],
                "statement_signature_sha256": None,
                "anomaly_score": None,
            }
        )
    expected = _expected_labels(condition, root_a, root_b)
    observed_atoms = {atom for row in observations for atom in row["atoms"]}
    evidence_checks = {
        root: REQUIRED_EVIDENCE_ATOM[root] in observed_atoms
        for root in expected
        if root in REQUIRED_EVIDENCE_ATOM
    }
    all_connections = [value for runner in runners for value in runner.connection_ids]
    recovery = _recovery_audit(config, all_connections)
    runner_rows = [runner.as_dict() for runner in runners]
    workload_errors = list(baseline_workload_errors) + [
        str(worker.failure) for worker in workers if worker.failure
    ]
    mechanism_passed = (
        all(row["execution_count"] > 0 and not row["errors"] for row in runner_rows)
        and not workload_errors
    )
    evidence_passed = all(evidence_checks.values())
    result = {
        "protocol": PROTOCOL,
        "case_id": case_id,
        "collected_at": _utc_now(),
        "collection": {
            "table": table,
            "replicate_index": replicate,
            "condition": condition,
            "schedule": [{"root": root, "offset_seconds": offset, "duration_seconds": root_duration} for root, offset in schedule],
            "baseline_seconds": baseline_seconds,
            "observation_seconds": observation_seconds,
            "recovery_seconds": recovery_seconds,
            "workload_worker_count": workload_count,
        },
        "truth": {"expected_labels": expected},
        "runner_audit": runner_rows,
        "workload_audit": {"errors": workload_errors, "passed": not workload_errors},
        "recovery_audit": recovery,
        "evidence_audit": {"by_root": evidence_checks, "observed_atoms": sorted(observed_atoms)},
        "status": "collected" if mechanism_passed and evidence_passed and recovery["passed"] else "failed_audit",
    }
    _write_json(case_root / "case.json", result)
    _write_json(case_root / "metrics.json", {"baseline": list(baseline_metrics), "anomaly": anomaly_metrics})
    _write_json(case_root / "statement_trace.json", {"baseline": baseline_trace, "anomaly": anomaly_trace, "selected_anomalous": selected})
    _write_json(case_root / "plans.json", plans)
    _write_json(case_root / "semantic_observations.json", {"observations": observations})
    if recovery_seconds > 1.0:
        time.sleep(recovery_seconds - 1.0)
    return result


def _environment_audit(config: MySQLConfig, workload_count: int) -> Dict[str, Any]:
    connection = config.connect()
    with connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION(), @@version_comment, @@performance_schema")
            version, comment, performance_schema = cursor.fetchone()
            cursor.execute(
                "SELECT TABLE_NAME, TABLE_ROWS FROM information_schema.tables "
                "WHERE TABLE_SCHEMA=%s AND TABLE_NAME IN ('customer','orders','order_line') ORDER BY TABLE_NAME",
                (config.database,),
            )
            tables = [{"table": row[0], "estimated_rows": int(row[1] or 0)} for row in cursor.fetchall()]
            cursor.execute("UPDATE performance_schema.setup_consumers SET ENABLED='YES' WHERE NAME='events_statements_history_long'")
    return {
        "mysql_version": str(version),
        "mysql_version_comment": str(comment),
        "performance_schema_enabled": bool(performance_schema),
        "database": config.database,
        "tables": tables,
        "workload_worker_count": workload_count,
        "credential_storage": "not persisted",
    }


def collect(args: argparse.Namespace) -> None:
    password = os.environ.get(args.password_env) or _container_password(args.container)
    config = MySQLConfig(args.host, args.port, args.user, password, args.database)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": PROTOCOL,
        "created_at": _utc_now(),
        "seed": args.seed,
        "replicate_count": args.replicates,
        "tables": list(args.tables),
        "root_pairs": [list(row) for row in PAIR_ROOTS],
        "conditions": list(args.conditions),
        "timing": {
            "baseline_seconds": args.baseline_seconds,
            "observation_seconds": args.observation_seconds,
            "root_duration_seconds": args.root_duration_seconds,
            "overlap_seconds": args.root_duration_seconds / 2.0,
            "recovery_seconds": args.recovery_seconds,
            "sample_interval_seconds": 1.0,
        },
        "metric_schema": {
            "metrics": list(METRICS),
            "time_bins_seconds": [list(row) for row in TIME_BINS_SECONDS],
        },
        "evidence_policy": {
            "statement_selection": "phase-rate delta over instrumented runtime statements",
            "semantic_extraction_inputs": ["selected_statement", "EXPLAIN FORMAT=JSON", "runtime lock-wait observation"],
            "forbidden_inputs": ["root label", "scenario", "condition", "case ID"],
        },
        "environment": _environment_audit(config, args.workload_workers),
    }
    _write_json(output / "protocol.json", protocol)

    requested = []
    for replicate in range(1, args.replicates + 1):
        for table in args.tables:
            for root_a, root_b in PAIR_ROOTS:
                for condition in args.conditions:
                    requested.append((table, root_a, root_b, replicate, condition))
    if args.max_cases:
        requested = requested[: args.max_cases]
    completed = 0
    failed = 0
    for table, root_a, root_b, replicate, condition in requested:
        case_id = _case_id(table, root_a, root_b, replicate, condition)
        case_path = output / "cases" / case_id / "case.json"
        if case_path.exists() and not args.overwrite:
            row = json.loads(case_path.read_text(encoding="utf-8"))
            if row.get("status") == "collected":
                completed += 1
                continue
        row: Dict[str, Any] = {}
        for attempt in range(1, args.max_attempts + 1):
            baseline_trace = StatementTrace()
            baseline_trace.set_phase("baseline")
            workers = [
                Workload(config, baseline_trace, args.seed + replicate * 100 + index)
                for index in range(args.workload_workers)
            ]
            for worker in workers:
                worker.start()
            baseline_reference = time.monotonic() + args.baseline_seconds
            baseline_metrics = _sample_for(config, args.baseline_seconds, baseline_reference)
            for worker in workers:
                worker.stop()
            for worker in workers:
                worker.join(timeout=10)
            row = _collect_case(
                config,
                output,
                table,
                root_a,
                root_b,
                replicate,
                condition,
                baseline_metrics,
                baseline_trace.snapshot("baseline"),
                [str(worker.failure) for worker in workers if worker.failure],
                args.baseline_seconds,
                args.observation_seconds,
                args.root_duration_seconds,
                args.recovery_seconds,
                args.workload_workers,
                args.seed + completed * 1000 + attempt,
            )
            row["collection_attempt"] = attempt
            _write_json(case_path, row)
            if row["status"] == "collected":
                break
        completed += int(row["status"] == "collected")
        failed += int(row["status"] != "collected")
        _write_json(
            output / "progress.json",
            {
                "protocol": PROTOCOL,
                "updated_at": _utc_now(),
                "requested_case_count": len(requested),
                "completed_case_count": completed,
                "failed_case_count": failed,
                "last_case_id": case_id,
                "last_status": row["status"],
            },
        )
        print(f"[{completed + failed}/{len(requested)}] {case_id}: {row['status']}", flush=True)
        if row["status"] != "collected" and args.fail_fast:
            raise RuntimeError(f"collection audit failed for {case_id}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--container", default="dbmags-mysql")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password-env", default="DBMAGS_MYSQL_PASSWORD")
    parser.add_argument("--database", default="tpcc10_test")
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--tables", nargs="+", default=list(TABLES), choices=TABLES)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--baseline-seconds", type=float, default=60.0)
    parser.add_argument("--observation-seconds", type=float, default=75.0)
    parser.add_argument("--root-duration-seconds", type=float, default=30.0)
    parser.add_argument("--recovery-seconds", type=float, default=5.0)
    parser.add_argument("--workload-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.replicates < 1 or args.workload_workers < 1 or args.max_attempts < 1:
        parser.error("replicates, workload workers, and max attempts must be positive")
    if min(args.baseline_seconds, args.observation_seconds, args.root_duration_seconds, args.recovery_seconds) <= 0:
        parser.error("all durations must be positive")
    if args.observation_seconds < 2.0 * args.root_duration_seconds:
        parser.error("observation duration must cover both sequential injections")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    collect(parse_args(argv))


if __name__ == "__main__":
    main()
