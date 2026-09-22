"""SQLite 存储层（标准库 sqlite3，线程安全封装）。"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "detections.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                    TEXT PRIMARY KEY,
    series_id             TEXT,
    created_at            TEXT NOT NULL,
    request_params        TEXT NOT NULL,
    result_json           TEXT NOT NULL,
    point_count           INTEGER NOT NULL,
    anomaly_segment_count INTEGER NOT NULL,
    anomaly_point_count   INTEGER NOT NULL,
    threshold             REAL NOT NULL,
    window_size           INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_series_id ON detections(series_id);
CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections(created_at);

CREATE TABLE IF NOT EXISTS segment_statuses (
    detection_id  TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open',
    note          TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (detection_id, segment_index)
);
CREATE INDEX IF NOT EXISTS idx_segment_statuses_detection
    ON segment_statuses(detection_id);
"""


def _backfill_segment_statuses(conn: sqlite3.Connection) -> None:
    """为存量检测补写默认 open 的处置记录（幂等，可重复执行）。

    segment_statuses 表后于 detections 引入：升级已有数据库时，
    按 result_json 中的异常段数为每个历史段补一行 open 记录，
    updated_at 取检测的 created_at；已存在的行保持不变。
    """
    rows = conn.execute("SELECT id, created_at, result_json FROM detections").fetchall()
    for det_id, created_at, result_json in rows:
        segments = json.loads(result_json).get("segments", [])
        conn.executemany(
            """
            INSERT OR IGNORE INTO segment_statuses
                (detection_id, segment_index, status, note, updated_at)
            VALUES (?, ?, 'open', NULL, ?)
            """,
            [(det_id, idx, created_at) for idx in range(len(segments))],
        )


def init_db(db_path: str | None = None) -> None:
    """建库建表并启用 WAL；对存量检测回填默认处置记录。"""
    db_path = db_path or DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        _backfill_segment_statuses(conn)


@contextmanager
def get_conn(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path or DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=15000")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_detection(
    detection_id: str,
    series_id: str | None,
    request_params: dict[str, Any],
    result: dict[str, Any],
    db_path: str | None = None,
) -> str:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO detections (
                id, series_id, created_at, request_params, result_json,
                point_count, anomaly_segment_count, anomaly_point_count,
                threshold, window_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detection_id,
                series_id,
                created_at,
                json.dumps(request_params, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                result["point_count"],
                result["anomaly_segment_count"],
                result["anomaly_point_count"],
                result["threshold"],
                result["window_size"],
            ),
        )
        # 同一事务为每个异常段写入默认 open 处置记录（updated_at 取检测创建时间）
        conn.executemany(
            """
            INSERT INTO segment_statuses
                (detection_id, segment_index, status, note, updated_at)
            VALUES (?, ?, 'open', NULL, ?)
            """,
            [
                (detection_id, idx, created_at)
                for idx in range(len(result.get("segments", [])))
            ],
        )
    return created_at


def get_detection(detection_id: str, db_path: str | None = None) -> dict[str, Any] | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "series_id": row["series_id"],
        "created_at": row["created_at"],
        "request_params": json.loads(row["request_params"]),
        "result": json.loads(row["result_json"]),
    }


def get_segment_statuses(
    detection_id: str, db_path: str | None = None
) -> list[dict[str, Any]]:
    """按段序号升序返回一条检测全部异常段的处置记录（可能为空列表）。"""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT segment_index, status, note, updated_at
            FROM segment_statuses WHERE detection_id = ?
            ORDER BY segment_index
            """,
            (detection_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def transition_segment_status(
    detection_id: str,
    segment_index: int,
    target: str,
    note: str | None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """对单个异常段执行一次处置状态提交（读-判-写同一事务，BEGIN IMMEDIATE 防并发交错）。

    返回 ``{"outcome": ..., "record": {...}}``：

    - ``outcome="transition"``：合法前进一步，状态已推进，updated_at 刷新；
      请求带非空 note 时覆盖旧 note，否则保留；
    - ``outcome="idempotent"``：幂等重放（目标 == 当前），状态不变；
      仅当请求带非空 note 时更新 note 与 updated_at；
    - ``outcome="conflict"``：非法跳转，不做任何写入。

    调用方需先确认检测与段序号存在；记录缺失时抛 KeyError。
    """
    from . import segments as seg

    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path or DB_PATH, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                """
                SELECT status, note, updated_at FROM segment_statuses
                WHERE detection_id = ? AND segment_index = ?
                """,
                (detection_id, segment_index),
            ).fetchone()
            if row is None:
                raise KeyError(f"处置记录不存在: {detection_id}#{segment_index}")

            outcome = seg.decide(row["status"], target)
            if outcome == seg.OUTCOME_TRANSITION:
                new_note = note if note is not None else row["note"]
                conn.execute(
                    """
                    UPDATE segment_statuses
                    SET status = ?, note = ?, updated_at = ?
                    WHERE detection_id = ? AND segment_index = ?
                    """,
                    (target, new_note, now, detection_id, segment_index),
                )
                record = {"status": target, "note": new_note, "updated_at": now}
            elif outcome == seg.OUTCOME_IDEMPOTENT and note is not None:
                conn.execute(
                    """
                    UPDATE segment_statuses
                    SET note = ?, updated_at = ?
                    WHERE detection_id = ? AND segment_index = ?
                    """,
                    (note, now, detection_id, segment_index),
                )
                record = {"status": row["status"], "note": note, "updated_at": now}
            else:
                record = {
                    "status": row["status"],
                    "note": row["note"],
                    "updated_at": row["updated_at"],
                }
            conn.execute("COMMIT")
            return {"outcome": outcome, "record": record}
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def get_status_counts(
    detection_ids: list[str], db_path: str | None = None
) -> dict[str, dict[str, int]]:
    """批量统计各检测的异常段处置状态计数（无记录的检测不在返回中）。"""
    if not detection_ids:
        return {}
    placeholders = ",".join("?" for _ in detection_ids)
    with get_conn(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT detection_id, status, COUNT(*) AS cnt
            FROM segment_statuses
            WHERE detection_id IN ({placeholders})
            GROUP BY detection_id, status
            """,
            detection_ids,
        ).fetchall()
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = counts.setdefault(
            row["detection_id"], {"open": 0, "acknowledged": 0, "resolved": 0}
        )
        bucket[row["status"]] = row["cnt"]
    return counts


def list_detections(
    series_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db_path: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    with get_conn(db_path) as conn:
        if series_id is not None:
            total = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE series_id = ?", (series_id,)
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT id, series_id, created_at, point_count,
                       anomaly_segment_count, anomaly_point_count,
                       threshold, window_size
                FROM detections WHERE series_id = ?
                ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (series_id, limit, offset),
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            rows = conn.execute(
                """
                SELECT id, series_id, created_at, point_count,
                       anomaly_segment_count, anomaly_point_count,
                       threshold, window_size
                FROM detections ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
    items = [dict(r) for r in rows]
    counts = get_status_counts([item["id"] for item in items], db_path)
    for item in items:
        item["status_counts"] = counts.get(
            item["id"], {"open": 0, "acknowledged": 0, "resolved": 0}
        )
    return items, total
