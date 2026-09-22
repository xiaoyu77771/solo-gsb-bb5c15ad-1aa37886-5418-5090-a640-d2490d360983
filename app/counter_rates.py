"""无状态计数器速率计算：回卷 / 重置 / 断档感知的累计读数差分。

输入带时间戳的**累计计数读数**（单调递增计数器的快照序列），逐相邻区间
输出增量与每秒速率，并处理三类边界：

- **回卷（wrap）**：计数器到达 ``max_counter`` 后归零重新计数。负增量在
  「前值距 ``max_counter``、后值距 0 均不超过 ``wrap_tolerance``」时按
  回卷计算，增量为 ``(max_counter - prev) + 1 + curr``（取值域
  ``[0, max_counter]``，与 SNMP 32/64 位计数器惯例一致）；
- **重置（reset）**：不满足回卷条件的负增量（包括未提供 ``max_counter``
  时的一切负增量），判定为计数器重置边界，区间增量不计入汇总；
- **未知（unknown）**：区间时间跨度超过 ``max_gap_seconds`` 时，期间计数
  变化不可知（可能发生多次回卷 / 重置），区间不计入汇总。

reset 与 unknown 区间都会**切断连续 epoch**：epoch 是一串连续的计入区间
（normal / wrap），每个 epoch 输出累计增量、时长与平均速率；顶层汇总值
同样只统计计入区间，排除 reset 与 unknown。接口无状态，结果不写入
SQLite。
"""

from __future__ import annotations

from .models import parse_timestamp
from .resampling import epoch_microseconds, inclusive_gap_microseconds

_MICROSECONDS_PER_SECOND = 1_000_000


def compute_counter_rates(
    timestamps: list[str],
    values: list[float],
    *,
    max_counter: float | None,
    wrap_tolerance: float,
    max_gap_seconds: float | None,
) -> dict:
    """逐区间差分累计计数读数，判定 normal / wrap / reset / unknown 并汇总 epoch。

    ``timestamps`` 必须已按时间排序且无重复（由接口层保证），``values``
    均为有限数值。判定优先级：unknown（跨度超限）> normal（增量非负）>
    wrap（负增量且两侧均在容差内）> reset（其余负增量）。
    """
    sample_us = [epoch_microseconds(parse_timestamp(ts)) for ts in timestamps]
    gap_us = (
        inclusive_gap_microseconds(max_gap_seconds)
        if max_gap_seconds is not None
        else None
    )

    intervals: list[dict] = []
    epochs: list[dict] = []
    run: list[dict] = []  # 当前连续 epoch 的计入区间缓冲
    counts = {"normal": 0, "wrap": 0, "reset": 0, "unknown": 0}
    total_delta = 0.0
    counted_duration_us = 0

    def close_epoch() -> None:
        """收尾当前连续 epoch：汇总计入区间的增量 / 时长 / 平均速率。"""
        nonlocal run
        if not run:
            return
        first, last = run[0], run[-1]
        epoch_delta = sum(iv["delta"] for iv in run)
        epoch_duration = (
            sum(iv["_duration_us"] for iv in run) / _MICROSECONDS_PER_SECOND
        )
        epochs.append(
            {
                "epoch_index": len(epochs),
                "start_index": first["start_index"],
                "end_index": last["end_index"],
                "start_time": first["start_time"],
                "end_time": last["end_time"],
                "interval_count": len(run),
                "total_delta": epoch_delta,
                "duration_seconds": epoch_duration,
                "average_rate_per_second": (
                    epoch_delta / epoch_duration if epoch_duration > 0 else None
                ),
            }
        )
        run = []

    for i in range(len(values) - 1):
        prev, curr = values[i], values[i + 1]
        dt_us = sample_us[i + 1] - sample_us[i]  # 重复时间戳已被接口层拒绝，恒 > 0
        duration = dt_us / _MICROSECONDS_PER_SECOND
        raw = curr - prev

        delta: float | None
        if gap_us is not None and dt_us > gap_us:
            classification = "unknown"
            delta = None
            reason = (
                f"区间跨度 {duration} 秒超过 max_gap_seconds={max_gap_seconds}，"
                "期间计数变化不可知，标记为 unknown 且不计入汇总"
            )
        elif raw >= 0:
            classification = "normal"
            delta = raw
            reason = f"增量 {raw} 非负，按实际差值计入"
        elif (
            max_counter is not None
            and abs(max_counter - prev) <= wrap_tolerance
            and abs(curr) <= wrap_tolerance
        ):
            classification = "wrap"
            delta = (max_counter - prev) + 1.0 + curr
            reason = (
                f"前值 {prev} 距 max_counter={max_counter}、后值 {curr} 距 0 "
                f"均不超过容差 {wrap_tolerance}，按回卷计入："
                f"({max_counter}-{prev})+1+{curr}={delta}"
            )
        else:
            classification = "reset"
            delta = None
            if max_counter is None:
                reason = (
                    f"增量 {raw} 为负且未提供 max_counter，无法按回卷处理，"
                    "判定为计数器重置边界，不计入汇总"
                )
            else:
                reason = (
                    f"增量 {raw} 为负，但前值 {prev} 距 max_counter={max_counter} "
                    f"或后值 {curr} 距 0 超过容差 {wrap_tolerance}，"
                    "判定为计数器重置边界，不计入汇总"
                )

        counts[classification] += 1
        counted = delta is not None
        if counted:
            total_delta += delta
            counted_duration_us += dt_us
        else:
            close_epoch()  # reset / unknown 切断连续 epoch

        interval = {
            "index": i,
            "start_index": i,
            "end_index": i + 1,
            "start_time": timestamps[i],
            "end_time": timestamps[i + 1],
            "start_value": prev,
            "end_value": curr,
            "duration_seconds": duration,
            "raw_delta": raw,
            "delta": delta,
            "rate_per_second": (delta / duration) if counted else None,
            "classification": classification,
            "reason": reason,
            "epoch_index": len(epochs) if counted else None,
            "_duration_us": dt_us,
        }
        intervals.append(interval)
        if counted:
            run.append(interval)

    close_epoch()

    for iv in intervals:  # 内部字段不随响应返回
        del iv["_duration_us"]

    counted_duration = counted_duration_us / _MICROSECONDS_PER_SECOND
    return {
        "point_count": len(values),
        "max_counter": max_counter,
        "wrap_tolerance": wrap_tolerance,
        "max_gap_seconds": max_gap_seconds,
        "interval_count": len(intervals),
        "normal_count": counts["normal"],
        "wrap_count": counts["wrap"],
        "reset_count": counts["reset"],
        "unknown_count": counts["unknown"],
        "epoch_count": len(epochs),
        "total_delta": total_delta,
        "counted_duration_seconds": counted_duration,
        "average_rate_per_second": (
            total_delta / counted_duration if counted_duration > 0 else None
        ),
        "intervals": intervals,
        "epochs": epochs,
    }
