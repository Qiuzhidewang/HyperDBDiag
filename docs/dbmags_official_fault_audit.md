# DB-MAGS Official Fault Audit

Scope: upstream `third_party/db-mags-official` at commit
`54fab0144d4a280a2fa256f3d0462f1edd72df7f`, checked against the active
`dbmags-mysql` container on 2026-08-02. This is an implementation and
environment audit, not an experiment result.

## Label accounting

The upstream README says 18 minor fault types. The single-fault dispatcher
has 19 IDs because ID 18 is a composite (`lock + slow SQL`), not a nineteenth
atomic root. The 18 atomic roots are 3 lock roots, 7 SQL roots, dump, 5
resource roots, and 2 workload roots. Do not train or report ID 18 as one
atomic label.

Evidence: `single_anomaly.py:80-153`, `tpcc_operation_set.py:89-99`, and
`multi_anomaly.py:41-66` in the upstream checkout.

## Environment facts

* MySQL 8.0.46, `performance_schema=ON`; `metadata_locks`, `data_locks`, and
  `data_lock_waits` are available. The metadata-lock instrument is enabled.
* Only `tpcc10_test` is present. The upstream `order_by_test` database,
  generated `table_*` data, and `/root/mysqlrc/fault_injection/Sql_Data/*`
  files are absent.
* `mysqldump` exists **inside** `dbmags-mysql`, not on the host running the
  Python collector. The image has shell utilities (`yes`, `dd`, `timeout`)
  but no Python, `stress-ng`, `tc`, or `ip`.
* The container has `cpu.max = 100000 100000` (one CPU quota despite
  `nproc=8`), no memory limit, and a writable Docker volume at
  `/var/lib/mysql`. `/tmp` is overlay storage, not that data volume.
* The effective capability set excludes `CAP_NET_ADMIN`. Consequently the
  official network-delay/loss mechanism cannot run here.

## Root-by-root contract

| Atomic root | Upstream mechanism and source | Current direct executability | Required repeatable evidence | Required cleanup |
| --- | --- | --- | --- | --- |
| `table_lock` | `LOCK TABLES <tpcc table> WRITE` (`tpcc_operation_set.py:101-108`) | Feasible on TPC-C tables | Record injector `CONNECTION_ID`; show its granted table/metadata lock and a separate bounded probe blocked by it | Always `UNLOCK TABLES` in `finally`, then close both sessions; verify no injector connection, transaction, metadata lock, or data-lock wait remains |
| `meta_data_lock` | `ALTER TABLE <tpcc table> ADD meta_data char(2)` (`tpcc_operation_set.py:110-117`) | Upstream form is not reusable: it changes schema and repeats fail on duplicate column | Use two sessions: holder has a transactional read; requester issues bounded DDL. Capture `metadata_locks` with one granted and one pending MDL plus expected timeout | Roll back/close holder; wait for or kill requester; prove no added column and no residual MDL. A successful DDL must be explicitly reversed |
| `record_lock` | Broad no-op update of a TPC-C indexed column (`tpcc_operation_set.py:119-127`) | Feasible | Persist SQL and injector ID; `performance_schema.data_locks` must show locks from that thread and a conflicting bounded update must wait | `ROLLBACK` and close injector/probe; verify its `innodb_trx`, `data_locks`, and `data_lock_waits` rows are gone |
| `missing_index` | Random predicate on listed non-indexed columns (`tpcc_operation_set.py:195-204`) | Feasible on TPC-C after per-SQL plan check | Persist exact SQL and `EXPLAIN FORMAT=JSON`; require target table and broad access (for example `ALL` with rows threshold) | Close/rollback statement session; retain plan and statement outcome |
| `too_much_index` | Updates one of four indexed columns in six `order_by_test.table_*` tables (`tpcc_operation_set.py:206-215`) | Not executable: database/tables absent | Dedicated fixture must record secondary-index count, write latency or redo/I/O amplification against a low-index control, and affected-row evidence | Roll back injection transaction or drop only a uniquely prefixed fixture after no active users remain |
| `implicit_conversion` | Join custom implicit-conversion table with `history` (`tpcc_operation_set.py:217-220`) | Not executable: referenced table is absent; upstream preparation creates `...table1`, not the queried name (`Preparation.py:55-62`) | Persist predicate and plan showing the conversion plus loss of selective indexed access/broad scan | Read-only session close; fixture teardown only if a dedicated table was created |
| `query_with_too_much_join` | Reads external generated SQL and rewrites it to `order_by_test` (`tpcc_operation_set.py:221-238`) | Not executable: file and schema absent | Persist plan with the required join count and per-root row/cardinality gate; enforce statement deadline | Cancel/close on deadline; no fixture state unless a dedicated dataset was created |
| `order_by` | Reads external generated SQL, intended to use disk temporary work (`tpcc_operation_set.py:240-257`) | Not executable directly: file/schema absent | Require `EXPLAIN` evidence of `Using filesort` and an observed sort/temp or statement-time increase. The local `customer ORDER BY c_last` fallback does show `Using filesort` | Close bounded read-only statement; retain plan/event evidence |
| `group_by` | Reads external generated SQL, intended to use disk temporary work (`tpcc_operation_set.py:259-276`) | Not executable directly: file/schema absent | Require grouping/aggregation plan and `Using temporary` or a measured `Created_tmp_*` delta. Current local fallback queries do **not** meet that gate on this schema | Close bounded read-only statement; retain plan/event evidence |
| `query_whole_table` | Reads six large `order_by_test` tables with `LIMIT 10000` (`tpcc_operation_set.py:278-286`) | Not executable directly: tables absent | Require plan `ALL` on the target and observed rows examined/statement event above a predeclared threshold; a `LIMIT` alone is not proof of a whole-table scan | Close/cancel bounded statement; no state mutation |
| `dump` | `mysqldump` to shared `test.sql`, with hard-coded credentials (`tpcc_operation_set.py:142-169`) | Upstream command is unusable here; container has `mysqldump` | Execute inside `dbmags-mysql`, record child PID, exit status, bounded output size, and MySQL statement/process evidence | Kill/wait child by deadline, remove only its unique output file, and verify no dump child remains |
| `cpu` | ChaosBlade CPU full-load (`tpcc_operation_set.py:171-174`) | Not executable: ChaosBlade absent | Run a bounded child in the MySQL container cgroup; record PID, elapsed time, `cpu.stat`/container CPU change, and MySQL telemetry change | Deadline/kill child and verify cgroup process absence. Host-side pressure is not a container CPU root |
| `io` | ChaosBlade disk burn on `/home` (`tpcc_operation_set.py:176-180`) | Not executable: ChaosBlade absent | Bounded read/write child on the same filesystem as MySQL (`/var/lib/mysql` volume, not host `/tmp`); record cgroup I/O counters and temporary path | Wait/kill child, `fsync` as appropriate, delete only its unique file, verify free-space floor |
| `disk` | ChaosBlade fills `/home` by 40 GB (`tpcc_operation_set.py:182-184`) | Not executable and unsafe | Treat as bounded disk-write pressure, not disk exhaustion; record free-space floor, bytes written, cgroup I/O delta, and MySQL impact | Stop before floor, delete only unique pressure file, verify free space and no child. Never reproduce the 40 GB fill on this artifact |
| `mem` | ChaosBlade 70% memory load (`tpcc_operation_set.py:186-188`) | Not executable: ChaosBlade absent; container has no memory cap | Bounded allocation process inside container with explicit cap; record `memory.current`, `memory.events`, allocator result, and MySQL telemetry | Ensure process exits, no OOM event/kill, and memory returns near baseline |
| `net` | ChaosBlade `network delay/loss` on `eth0`, MySQL port 3306 (`tpcc_operation_set.py:190-193`) | Not executable: no ChaosBlade, `tc`, `ip`, or `CAP_NET_ADMIN` | Only accept an impairment of the actual MySQL TCP path plus before/during/after query-latency evidence and reversal evidence | Delete qdisc/rule in `finally`, then verify normal path. Local loopback UDP load is not a network fault |
| `flow_workload` | Sets module-local carrier range and injects one slow predicate (`single_anomaly.py:137-144`, `179-188`) | Not valid as-is: `doOne()` reads `Tpcc.tpcc.MIN_CARRIER_ID`, not the module-local values (`Tpcc/tpcc.py:526-562`) | Controlled client rate with fixed mix; record attempted/completed transaction counts and `Questions`/`Com_*` deltas per window | Stop/join every worker, close all client sessions, restore rate/mix |
| `traffic_spike` | Increases runner ceiling from 300 to 500 and sets sleep to `0.044` (`single_anomaly.py:152-153`, `252-258`) | Needs a replacement executor; default README sleep is already `0.044` | Same fixed-mix rate/concurrency evidence as `flow_workload`; prove measured rate/concurrency changed rather than inferring it from configuration | Stop/join workers, close sessions, restore baseline throttle/concurrency |

## Composite ID 18

ID 18 calls `lock_slow_query()` (`single_anomaly.py:145-151`). That helper
constructs `LOCK TABLES ... WRITE` plus a count query
(`tpcc_operation_set.py:89-99`). It is therefore a two-root temporal case,
not `table_lock+slow_sql` as an atomic label. Validate and clean it as a
two-session table-lock/slow-query interaction; retain both component labels
and their planned/observed overlap.

## Upstream runner limitations

The official `Connection` hard-codes an external host and credentials
(`Connection/Connection.py:7-16,55-60`), its direct script entry point ignores
CLI arguments in favor of a hard-coded long run (`single_anomaly.py:349-360`),
and `Fault_Session` commits after five seconds without a `finally` cleanup
(`single_anomaly.py:287-293`). It is a source of fault definitions, not a
safe portable collector.

The removed local compatibility collector changed several upstream mechanisms
and is not part of the active artifact. Only the frozen v10 cohort and the
upstream source audit remain.
