import contextvars
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


EVALUATION_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
TRIAL_FLUSH_BATCH_SIZE = 1000

HARDWARE_COLUMNS = (
    "device_count",
    "link_count_per_device",
    "main_memory_gb",
    "global_buffer_mb",
    "core_count",
    "local_buffer_kb",
    "lane_count",
    "array_height",
    "vector_width",
)
TASK_COLUMNS = (
    "batch_size",
    "d_model",
    "n_heads",
    "input_seq_len",
    "data_type",
    "n_layers",
)
MATMUL_OPERATORS = (
    "QKV_proj",
    "Q_mul_K",
    "A_mul_V",
    "H_matmul0",
    "H_matmul1",
    "H_matmul2",
)
MATMUL_RESULT_FIELDS = (
    ("mapping_source", "TEXT NOT NULL"),
    ("execution_kind", "TEXT NOT NULL"),
    ("strategy", "TEXT"),
    ("M", "INTEGER"),
    ("N", "INTEGER"),
    ("K", "INTEGER"),
    ("l2_tile_M", "INTEGER"),
    ("l2_tile_N", "INTEGER"),
    ("l2_tile_K", "INTEGER"),
    ("is_l2_double_buffering", "INTEGER"),
    ("l1_tile_M", "INTEGER"),
    ("l1_tile_N", "INTEGER"),
    ("l1_tile_K", "INTEGER"),
    ("l2_loop_order", "TEXT"),
    ("l1_loop_order", "TEXT"),
    ("l0_M_tiling_factor", "INTEGER"),
    ("l0_N_tiling_factor", "INTEGER"),
    ("l0_K_tiling_factor", "INTEGER"),
    ("dataflow", "TEXT"),
    ("local_latency_s", "REAL"),
)
SOFTMAX_RESULT_FIELDS = (
    ("mapping_source", "TEXT NOT NULL"),
    ("M", "INTEGER"),
    ("N", "INTEGER"),
    ("l2_tile_M", "INTEGER"),
    ("l2_tile_N", "INTEGER"),
    ("is_l2_double_buffering", "INTEGER"),
    ("l1_tile_M", "INTEGER"),
    ("l1_tile_N", "INTEGER"),
    ("is_l1_double_buffering", "INTEGER"),
    ("local_latency_s", "REAL"),
)
LAYERNORM_OPERATORS = ("layer_norm",)
LAYERNORM_RESULT_FIELDS = (
    ("mapping_source", "TEXT NOT NULL"),
    ("M", "INTEGER"),
    ("N", "INTEGER"),
    ("l2_tile_M", "INTEGER"),
    ("l2_tile_N", "INTEGER"),
    ("l1_tile_M", "INTEGER"),
    ("l1_tile_N", "INTEGER"),
    ("local_latency_s", "REAL"),
)

TRIAL_COLUMNS = (
    "candidate_ordinal",
    "operator_name",
    "operator_type",
    "execution_kind",
    "strategy",
    "M",
    "N",
    "K",
    "l2_tile_M",
    "l2_tile_N",
    "l2_tile_K",
    "is_l2_double_buffering",
    "l1_tile_M",
    "l1_tile_N",
    "l1_tile_K",
    "is_l1_double_buffering",
    "l2_loop_order",
    "l1_loop_order",
    "l0_M_tiling_factor",
    "l0_N_tiling_factor",
    "l0_K_tiling_factor",
    "dataflow",
    "cycle_count",
    "raw_local_latency_s",
    "latency_multiplier",
    "latency_additive_s",
    "local_latency_s",
    "mapping_json",
)

_ACTIVE_RECORDER = contextvars.ContextVar("active_design_point_recorder", default=None)
_TRIAL_LATENCY = contextvars.ContextVar(
    "trial_latency_modifiers",
    default={"strategy": None, "multiplier": 1.0, "additive_s": 0.0},
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return int(value)
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_to_scalar)


def _mapping_to_dict(mapping: Any) -> Dict[str, Any]:
    if mapping is None:
        return {}
    if isinstance(mapping, dict):
        return {key: _to_scalar(value) for key, value in mapping.items()}
    return {
        key: _to_scalar(value)
        for key, value in vars(mapping).items()
        if not key.startswith("_")
    }


def make_trials_table_name(evaluation_id: str) -> str:
    if not isinstance(evaluation_id, str) or not EVALUATION_ID_PATTERN.fullmatch(
        evaluation_id
    ):
        raise ValueError("evaluation_id must be 32 lowercase hexadecimal characters.")
    return f"operator_mapping_trials_{evaluation_id}"


def _main_result_columns():
    columns = []
    for operator in MATMUL_OPERATORS:
        columns.extend(
            (f"{operator}_{field}", sql_type) for field, sql_type in MATMUL_RESULT_FIELDS
        )
    columns.extend(
        (f"A_softmax_{field}", sql_type)
        for field, sql_type in SOFTMAX_RESULT_FIELDS
    )
    for operator in LAYERNORM_OPERATORS:
        columns.extend(
            (f"{operator}_{field}", sql_type)
            for field, sql_type in LAYERNORM_RESULT_FIELDS
        )
    return tuple(columns)


MAIN_RESULT_COLUMNS = _main_result_columns()


@dataclass(frozen=True)
class RecordingContext:
    database_path: str
    evaluation_id: str
    hardware_point_hash: str
    hardware_point: dict
    stage: str
    task: dict
    compile_mode: str
    total_area_mm2: float
    enabled: bool = True
    record_operator_trials: bool = True


class DesignPointRecorder:
    def __init__(self):
        self.context: Optional[RecordingContext] = None
        self.connection: Optional[sqlite3.Connection] = None
        self.trials_table_name: Optional[str] = None
        self._trial_batch = []
        self._candidate_ordinal = 0
        self._finalized = False

    def begin_stage(self, context: RecordingContext) -> None:
        if self.context is not None:
            raise RuntimeError("Recorder stage has already begun.")
        self.context = context
        if not context.enabled:
            return
        if context.stage not in ("prefill", "decode"):
            raise ValueError("Recording stage must be 'prefill' or 'decode'.")
        self.trials_table_name = make_trials_table_name(context.evaluation_id)
        self.connection = sqlite3.connect(context.database_path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self._create_fixed_schema()
        self._create_trials_table()
        self._write_metadata()
        self.connection.commit()

    def _create_fixed_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        columns = [
            "design_point_id INTEGER PRIMARY KEY AUTOINCREMENT",
            "evaluation_id TEXT NOT NULL UNIQUE",
            "hardware_point_hash TEXT NOT NULL",
            "trials_table_name TEXT NOT NULL UNIQUE",
            "stage TEXT NOT NULL",
            "compile_mode TEXT NOT NULL",
            "batch_size INTEGER NOT NULL",
            "d_model INTEGER NOT NULL",
            "n_heads INTEGER NOT NULL",
            "input_seq_len INTEGER NOT NULL",
            "data_type TEXT NOT NULL",
            "n_layers INTEGER NOT NULL",
            "latency_s REAL NOT NULL",
            "total_area_mm2 REAL NOT NULL",
            "area_mm2_compute REAL",
            "area_mm2_io REAL",
        ]
        columns.extend(f"{name} INTEGER NOT NULL" for name in HARDWARE_COLUMNS)
        columns.extend(f"{name} {sql_type}" for name, sql_type in MAIN_RESULT_COLUMNS)
        columns.extend(
            (
                "hardware_json TEXT NOT NULL",
                "best_software_mapping_json TEXT NOT NULL",
            )
        )
        self.connection.execute(
            f"CREATE TABLE IF NOT EXISTS hwsw_design_points "
            f"({', '.join(columns)})"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_design_points_hardware_hash "
            "ON hwsw_design_points(hardware_point_hash)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_design_points_latency "
            "ON hwsw_design_points(latency_s)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_design_points_area "
            "ON hwsw_design_points(total_area_mm2)"
        )

    def _create_trials_table(self) -> None:
        table = self.trials_table_name
        self.connection.execute(
            f"""
            CREATE TABLE {table} (
                trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_ordinal INTEGER NOT NULL UNIQUE,
                operator_name TEXT NOT NULL,
                operator_type TEXT NOT NULL,
                execution_kind TEXT NOT NULL,
                strategy TEXT,
                M INTEGER,
                N INTEGER,
                K INTEGER,
                l2_tile_M INTEGER,
                l2_tile_N INTEGER,
                l2_tile_K INTEGER,
                is_l2_double_buffering INTEGER,
                l1_tile_M INTEGER,
                l1_tile_N INTEGER,
                l1_tile_K INTEGER,
                is_l1_double_buffering INTEGER,
                l2_loop_order TEXT,
                l1_loop_order TEXT,
                l0_M_tiling_factor INTEGER,
                l0_N_tiling_factor INTEGER,
                l0_K_tiling_factor INTEGER,
                dataflow TEXT,
                cycle_count INTEGER,
                raw_local_latency_s REAL NOT NULL,
                latency_multiplier REAL NOT NULL DEFAULT 1.0,
                latency_additive_s REAL NOT NULL DEFAULT 0.0,
                local_latency_s REAL NOT NULL,
                mapping_json TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            f"CREATE INDEX {table}_operator_idx ON {table}(operator_name)"
        )
        self.connection.execute(
            f"CREATE INDEX {table}_latency_idx ON {table}(operator_name, local_latency_s)"
        )

    def _write_metadata(self) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO schema_metadata(key, value) VALUES (?, ?)",
            (
                ("created_by", "dse-framework-llmcompass-probe"),
                ("stage", self.context.stage),
            ),
        )

    def record_operator_mapping_trial(
        self,
        operator_name: str,
        operator_type: str,
        execution_kind: str,
        graph: Optional[Dict[str, Any]] = None,
        mapping: Any = None,
        cycle_count: Optional[int] = None,
        raw_local_latency_s: float = 0.0,
    ) -> None:
        if not self.context.enabled or not self.context.record_operator_trials:
            return
        graph = graph or {}
        mapping_dict = _mapping_to_dict(mapping)
        modifiers = _TRIAL_LATENCY.get()
        multiplier = float(modifiers["multiplier"])
        additive_s = float(modifiers["additive_s"])
        raw_latency = float(raw_local_latency_s)
        row = {
            "candidate_ordinal": self._candidate_ordinal,
            "operator_name": operator_name,
            "operator_type": operator_type,
            "execution_kind": execution_kind,
            "strategy": modifiers["strategy"],
            "M": graph.get("M"),
            "N": graph.get("N"),
            "K": graph.get("K"),
            "cycle_count": cycle_count,
            "raw_local_latency_s": raw_latency,
            "latency_multiplier": multiplier,
            "latency_additive_s": additive_s,
            "local_latency_s": raw_latency * multiplier + additive_s,
            "mapping_json": _stable_json(mapping_dict),
        }
        for column in TRIAL_COLUMNS:
            if column not in row:
                row[column] = mapping_dict.get(column)
            row[column] = _to_scalar(row[column])
        self._candidate_ordinal += 1
        self._trial_batch.append(tuple(row[column] for column in TRIAL_COLUMNS))
        if len(self._trial_batch) >= TRIAL_FLUSH_BATCH_SIZE:
            self.flush_trials()

    def flush_trials(self) -> None:
        if not self.context.enabled or not self._trial_batch:
            return
        placeholders = ", ".join("?" for _ in TRIAL_COLUMNS)
        self.connection.executemany(
            f"INSERT INTO {self.trials_table_name} "
            f"({', '.join(TRIAL_COLUMNS)}) VALUES ({placeholders})",
            self._trial_batch,
        )
        self.connection.commit()
        self._trial_batch.clear()

    def finalize_design_point(
        self,
        latency_s: float,
        best_software_mapping: Dict[str, Any],
        flattened_mapping: Dict[str, Any],
        area_mm2_compute: Optional[float] = None,
        area_mm2_io: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.context.enabled:
            return {}
        self.flush_trials()
        design_payload = {
            "stage": self.context.stage,
            "compile_mode": self.context.compile_mode,
            "task": self.context.task,
            "hardware_point": self.context.hardware_point,
            "best_software_mapping": best_software_mapping,
        }
        row = {
            "evaluation_id": self.context.evaluation_id,
            "hardware_point_hash": self.context.hardware_point_hash,
            "trials_table_name": self.trials_table_name,
            "stage": self.context.stage,
            "compile_mode": self.context.compile_mode,
            "latency_s": float(latency_s),
            "total_area_mm2": float(self.context.total_area_mm2),
            "area_mm2_compute": area_mm2_compute,
            "area_mm2_io": area_mm2_io,
            "hardware_json": _stable_json(self.context.hardware_point),
            "best_software_mapping_json": _stable_json(best_software_mapping),
        }
        row.update(self.context.task)
        row.update(self.context.hardware_point)
        row.update(flattened_mapping)
        columns = (
            "evaluation_id",
            "hardware_point_hash",
            "trials_table_name",
            "stage",
            "compile_mode",
            *TASK_COLUMNS,
            "latency_s",
            "total_area_mm2",
            "area_mm2_compute",
            "area_mm2_io",
            *HARDWARE_COLUMNS,
            *(name for name, _ in MAIN_RESULT_COLUMNS),
            "hardware_json",
            "best_software_mapping_json",
        )
        missing = [column for column in columns if column not in row]
        if missing:
            raise ValueError(f"Design-point row is missing columns: {missing}")
        placeholders = ", ".join("?" for _ in columns)
        self.connection.execute(
            f"INSERT INTO hwsw_design_points "
            f"({', '.join(columns)}) VALUES ({placeholders})",
            tuple(_to_scalar(row[column]) for column in columns),
        )
        self.connection.commit()
        self._finalized = True
        return {column: row[column] for column in columns}

    def abort_stage(self) -> None:
        self._trial_batch.clear()
        if (
            self.connection is None
            or self.trials_table_name is None
            or self._finalized
        ):
            return
        try:
            self.connection.rollback()
            self.connection.execute(f"DROP TABLE IF EXISTS {self.trials_table_name}")
            self.connection.commit()
        except sqlite3.Error:
            database_path = self.context.database_path
            self.close()
            try:
                connection = sqlite3.connect(database_path, timeout=30)
                connection.execute(f"DROP TABLE IF EXISTS {self.trials_table_name}")
                connection.commit()
                connection.close()
            except sqlite3.Error:
                pass

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None


@contextmanager
def recording_scope(recorder: DesignPointRecorder, context: RecordingContext):
    if recorder.context is None:
        raise RuntimeError("begin_stage() must be called before recording_scope().")
    if recorder.context is not context:
        raise ValueError("recording_scope context does not match recorder context.")
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    except Exception:
        recorder.abort_stage()
        raise
    finally:
        _ACTIVE_RECORDER.reset(token)
        if not recorder._finalized:
            recorder.abort_stage()


def get_active_recorder() -> Optional[DesignPointRecorder]:
    return _ACTIVE_RECORDER.get()


def get_trial_latency_modifiers() -> Dict[str, Any]:
    """Return the effective strategy/multiplier for an emitted trial."""
    return dict(_TRIAL_LATENCY.get())


@contextmanager
def trial_latency_scope(
    strategy: Optional[str] = None,
    multiplier: float = 1.0,
    additive_s: float = 0.0,
):
    token = _TRIAL_LATENCY.set(
        {
            "strategy": strategy,
            "multiplier": float(multiplier),
            "additive_s": float(additive_s),
        }
    )
    try:
        yield
    finally:
        _TRIAL_LATENCY.reset(token)
