"""无状态累计计数器速率计算：单调增量 / 回卷 / 重置 / 断档 unknown。

输入按时间排序的累计读数序列，逐相邻区间（``points[i-1] -> points[i]``）判定：

- ``normal``：后值 ≥ 前值，增量即后值 − 前值；
- ``wrap``：出现负增量，且**前值距 ``max_counter``** 与**后值距 0** 均不超过
  ``wrap_tolerance`` 时按计数器回卷（rollover）计算，
  增量 = ``(max_counter − 前值) + 后值``；
- ``reset``：负增量但不满足回卷条件（或未提供 ``max_counter``），标注重置边界，
  该区间增量不计入并切断连续 epoch；
- ``unknown``：区间时间跨度超过 ``max_gap_seconds``，增量无法估计，
  同样不计入并切断连续 epoch（断档判定优先于回卷 / 重置判定）。

连续的 normal / wrap 区间组成一个 **epoch**（重置与 unknown 都会断开）；
汇总增量 / 速率只统计 normal / wrap 区间，排除重置与 unknown。
时间比较一律换算为 epoch 整数微秒，``max_gap_seconds`` 含等号语义
（跨度恰好等于阈值时仍为已知区间）。
"""

from __future__ import annotations

from .models import parse_timestamp
from .resampling import epoch_microseconds, inclusive_gap_microseconds

_MICROSECONDS_PER_SECOND = 1_000_000

# 区间判定原因代码（reason_detail 为对应中文解释）
REASON_NORMAL = "normal"
REASON_WRAP = "wrap"
REASON_RESET = "reset"
REASON_UNKNOWN = "unknown"
_EPOCH_BREAK_REASONS = (REASON_RESET, REASON_UNKNOWN)


def compute_counter_rates(
    timestamps: list[str],
    values: list[float],
    *,
    max_counter: float | None,
    wrap_tolerance: float,
    max_gap_seconds: float | None,
    series_id: str | None = None,
) -> dict:
    """计算逐区间增量 / 速率、epoch 累计与汇总。

    ``timestamps`` 必须已按时间排序且无重复（由接口层保证）。
    省略 ``max_counter`` 时不做回卷判定（负增量一律标 reset）；
    省略 ``max_gap_seconds`` 时不做断档判定。
    """
    max_gap_us = (
        inclusive_gap_microseconds(max_gap_seconds)
        if max_gap_seconds is not None
        else None
    )
    sample_us = [epoch_microseconds(parse_timestamp(ts)) for ts in timestamps]
    n = len(values)

    intervals: list[dict] = []
    epochs: list[dict] = []
    open_epoch: dict | None = None
    counts = {
        REASON_NORMAL: 0,
        REASON_WRAP: 0,
        REASON_RESET: 0,
        REASON_UNKNOWN: 0,
    }

    def close_epoch() -> None:
        nonlocal open_epoch
        if open_epoch is not None:
            epochs.append(open_epoch)
            open_epoch = None

    for i in range(1, n):
        prev, curr = values[i - 1], values[i]
        elapsed_us = sample_us[i] - sample_us[i - 1]
        elapsed_seconds = elapsed_us / _MICROSECONDS_PER_SECOND
        raw_delta = curr - prev

        if max_gap_us is not None and elapsed_us > max_gap_us:
            reason = REASON_UNKNOWN
            delta: float | None = None
            reason_detail = (
                f"未知区间：时间跨度 {elapsed_seconds:g}s 超过 "
                f"max_gap_seconds={max_gap_seconds:g}s，增量无法估计并切断连续 epoch"
            )
        elif raw_delta >= 0:
            reason = REASON_NORMAL
            delta = raw_delta
            reason_detail = (
                f"正常增量：后值 {curr:g} ≥ 前值 {prev:g}，"
                f"增量 = 后值 − 前值 = {delta:g}"
            )
        elif max_counter is not None:
            dist_to_max = max_counter - prev
            dist_to_zero = curr
            if (
                0 <= dist_to_max <= wrap_tolerance
                and 0 <= dist_to_zero <= wrap_tolerance
            ):
                reason = REASON_WRAP
                delta = (max_counter - prev) + curr
                reason_detail = (
                    f"计数器回卷：负增量 {raw_delta:g}，且前值距 max_counter "
                    f"({dist_to_max:g}) 与后值距 0 ({dist_to_zero:g}) 均不超过容差 "
                    f"{wrap_tolerance:g}，增量 = (max_counter − 前值) + 后值 = {delta:g}"
                )
            else:
                reason = REASON_RESET
                delta = None
                reason_detail = (
                    f"重置边界：负增量 {raw_delta:g} 不满足回卷条件（前值距 max_counter "
                    f"{dist_to_max:g}、后值距 0 {dist_to_zero:g}，容差 {wrap_tolerance:g}），"
                    "该区间增量不计入并切断连续 epoch"
                )
        else:
            reason = REASON_RESET
            delta = None
            reason_detail = (
                f"重置边界：出现负增量 {raw_delta:g} 但未提供 max_counter，"
                "无法判定回卷，该区间增量不计入并切断连续 epoch"
            )

        counts[reason] += 1

        if reason in _EPOCH_BREAK_REASONS:
            close_epoch()
            epoch_index: int | None = None
            rate_per_second: float | None = None
        else:
            if open_epoch is None:
                open_epoch = {
                    "epoch_index": len(epochs),
                    "start_index": i - 1,
                    "end_index": i,
                    "start_time": timestamps[i - 1],
                    "end_time": timestamps[i],
                    "point_count": 1,
                    "interval_count": 0,
                    "total_delta": 0.0,
                    "duration_us": 0,
                }
            epoch_index = open_epoch["epoch_index"]
            open_epoch["end_index"] = i
            open_epoch["end_time"] = timestamps[i]
            open_epoch["point_count"] += 1
            open_epoch["interval_count"] += 1
            open_epoch["total_delta"] += delta
            open_epoch["duration_us"] += elapsed_us
            rate_per_second = delta / elapsed_seconds

        intervals.append(
            {
                "index": i - 1,
                "start_index": i - 1,
                "end_index": i,
                "start_time": timestamps[i - 1],
                "end_time": timestamps[i],
                "elapsed_seconds": elapsed_seconds,
                "from_value": prev,
                "to_value": curr,
                "delta": delta,
                "rate_per_second": rate_per_second,
                "reason": reason,
                "reason_detail": reason_detail,
                "epoch_index": epoch_index,
            }
        )

    close_epoch()

    # epoch 收尾：换算时长与速率
    epoch_results: list[dict] = []
    total_delta = 0.0
    total_duration_us = 0
    for epoch in epochs:
        duration_seconds = epoch["duration_us"] / _MICROSECONDS_PER_SECOND
        total_delta += epoch["total_delta"]
        total_duration_us += epoch["duration_us"]
        epoch_results.append(
            {
                "epoch_index": epoch["epoch_index"],
                "start_index": epoch["start_index"],
                "end_index": epoch["end_index"],
                "start_time": epoch["start_time"],
                "end_time": epoch["end_time"],
                "point_count": epoch["point_count"],
                "interval_count": epoch["interval_count"],
                "total_delta": epoch["total_delta"],
                "duration_seconds": duration_seconds,
                "rate_per_second": epoch["total_delta"] / duration_seconds,
            }
        )

    total_duration_seconds = total_duration_us / _MICROSECONDS_PER_SECOND
    overall_rate = (
        total_delta / total_duration_seconds if total_duration_us > 0 else None
    )

    return {
        "series_id": series_id,
        "point_count": n,
        "interval_count": n - 1,
        "max_counter": max_counter,
        "wrap_tolerance": wrap_tolerance,
        "max_gap_seconds": max_gap_seconds,
        "normal_count": counts[REASON_NORMAL],
        "wrap_count": counts[REASON_WRAP],
        "reset_count": counts[REASON_RESET],
        "unknown_count": counts[REASON_UNKNOWN],
        "epoch_count": len(epoch_results),
        "total_delta": total_delta,
        "total_duration_seconds": total_duration_seconds,
        "rate_per_second": overall_rate,
        "intervals": intervals,
        "epochs": epoch_results,
    }
