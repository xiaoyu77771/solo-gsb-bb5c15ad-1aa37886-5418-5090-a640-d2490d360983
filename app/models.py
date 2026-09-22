"""请求 / 响应数据模型。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# 有限浮点数（拒绝 NaN / Infinity），用于时滞候选与容差等标量参数
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


def parse_timestamp(value: object) -> datetime:
    """把时间戳解析为 UTC aware datetime。

    支持：
    - 数字：按秒或毫秒 epoch 自动识别；
    - 字符串：ISO-8601（如 ``2026-09-20T10:00:00+08:00``，
      ``2026-09-20 10:00:00`` 也可）；无时区按 UTC 处理。
    """
    if isinstance(value, bool):
        raise ValueError("时间戳不能是布尔值")
    if isinstance(value, (int, float)):
        ts = float(value)
        if not math.isfinite(ts):
            raise ValueError("数字时间戳必须有限")
        if abs(ts) > 1e11:  # 大于约 5138 年的秒数，按毫秒处理
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"无法解析时间戳字符串: {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise ValueError(f"不支持的时间戳类型: {type(value).__name__}")


class PointIn(BaseModel):
    """单个输入观测点。"""

    timestamp: object = Field(description="时间戳：ISO-8601 字符串，或 epoch 秒/毫秒数字")
    value: float = Field(description="观测数值")

    @field_validator("value")
    @classmethod
    def _finite_value(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value 必须是有限数值（不接受 NaN/Infinity）")
        return v


class DetectRequest(BaseModel):
    """检测请求。"""

    series_id: str | None = Field(
        default=None,
        max_length=128,
        description="可选的业务序列标识，便于检索与去重",
    )
    points: list[PointIn] = Field(min_length=2, description="带时间戳的数值序列")
    window_size: int = Field(default=20, ge=2, le=10_000, description="滑动窗口大小")
    min_window: int = Field(
        default=5, ge=1, le=10_000, description="开始评估所需的最小基线点数"
    )
    threshold: float = Field(
        default=3.0, gt=0, le=100, description="稳健 z 分数阈值，|z| 超过即判为异常"
    )
    merge_gap: int = Field(
        default=2,
        ge=0,
        le=10_000,
        description="合并异常区间时允许桥接的连续正常点数",
    )
    baseline_policy: Literal["rolling", "guarded"] = Field(
        default="rolling",
        description=(
            "基线策略：rolling=所有历史点滑入基线（默认）；"
            "guarded=仅正常点进入基线，连续异常达到 rebase_after 时重建基线"
        ),
    )
    rebase_after: int | None = Field(
        default=None,
        ge=1,
        le=10_000,
        description=(
            "guarded 模式下连续异常达到该点数即重建基线（默认取 min_window，"
            "且不得小于 min_window）；rolling 模式忽略"
        ),
    )
    gap_reset_factor: float | None = Field(
        default=None,
        gt=1.0,
        le=10_000,
        description=(
            "采样断档感知倍数（>1，默认关闭）：以排序后相邻点正时间差的中位数"
            "估算常规采样间隔，间隔超过该倍数即划分新分段，每段独立预热与维护"
            "基线/连续异常计数/重建代次，异常区间不跨越断档合并；启用至少需要 3 个点"
        ),
    )
    include_points: bool = Field(
        default=False, description="响应是否包含逐点注解（存储中始终保留）"
    )

    @model_validator(mode="after")
    def _check_windows(self) -> "DetectRequest":
        if self.min_window > self.window_size:
            raise ValueError("min_window 不能大于 window_size")
        if len(self.points) <= self.min_window:
            raise ValueError(
                f"至少需要 {self.min_window + 1} 个点（min_window={self.min_window}），"
                f"当前只有 {len(self.points)} 个"
            )
        if self.baseline_policy == "guarded" and self.rebase_after is not None:
            if self.rebase_after < self.min_window:
                raise ValueError(
                    f"rebase_after 不能小于 min_window"
                    f"（rebase_after={self.rebase_after} < min_window={self.min_window}）"
                )
        if self.gap_reset_factor is not None and len(self.points) < 3:
            raise ValueError(
                f"启用 gap_reset_factor 至少需要 3 个点以估算常规采样间隔，"
                f"当前只有 {len(self.points)} 个"
            )
        return self


class TrendRequest(BaseModel):
    """趋势扫描请求。"""

    series_id: str | None = Field(
        default=None,
        max_length=128,
        description="可选的业务序列标识，随响应回显",
    )
    points: list[PointIn] = Field(min_length=2, description="带时间戳的数值序列")
    window_size: int = Field(
        default=10, ge=3, le=10_000, description="滑动窗口大小（点数，≥3）"
    )
    step_size: int = Field(
        default=5,
        ge=1,
        le=10_000,
        description=(
            "滑动步长（点数）：小于 window_size 时窗口重叠，等于时首尾相接，"
            "大于时窗口间留有空档（空档两侧的显著窗口不会合并）"
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Mann-Kendall 显著性水平，p_value ≤ alpha 才可能判为趋势",
    )
    min_slope: float = Field(
        default=0.0,
        ge=0,
        description="最小绝对斜率（值/秒），|slope| 达到该值才判为趋势；默认 0 不过滤",
    )
    include_windows: bool = Field(
        default=False,
        description="响应是否包含全部评估窗口（趋势区间的成员窗口始终返回）",
    )

    @model_validator(mode="after")
    def _check_points(self) -> "TrendRequest":
        if len(self.points) < self.window_size:
            raise ValueError(
                f"点数必须不少于 window_size（window_size={self.window_size}），"
                f"当前只有 {len(self.points)} 个"
            )
        return self


class TrendWindow(BaseModel):
    """单个滑动窗口的趋势检验结果。"""

    window_index: int = Field(description="窗口序号（按滑动顺序从 0 开始）")
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    point_count: int
    s: int = Field(description="Mann-Kendall S 统计量")
    var_s: float = Field(description="S 的方差（含重复值修正）")
    z: float = Field(description="连续性校正后的 Z 值")
    p_value: float = Field(description="双侧 p 值（正态近似）")
    slope: float = Field(description="Theil-Sen 斜率（值/秒，按实际时间差）")
    significant: bool = Field(description="是否显著趋势（p≤alpha 且 |slope|≥min_slope）")
    direction: Literal["上升", "下降", "平稳"]


class TrendSegment(BaseModel):
    """合并后的趋势区间。"""

    direction: Literal["上升", "下降"]
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    point_count: int = Field(description="区间覆盖的原始点数")
    window_count: int = Field(description="成员窗口数")
    min_p_value: float = Field(description="成员窗口中的最小 p 值")
    median_slope: float = Field(description="成员窗口斜率的中位数（值/秒）")
    windows: list[TrendWindow] = Field(description="成员窗口明细")


class TrendResponse(BaseModel):
    series_id: str | None = None
    point_count: int
    window_size: int
    step_size: int
    alpha: float
    min_slope: float
    window_count: int = Field(description="评估窗口总数")
    significant_window_count: int = Field(description="显著趋势窗口数")
    trend_segment_count: int = Field(description="合并后的趋势区间数")
    segments: list[TrendSegment]
    windows: list[TrendWindow] | None = None

    model_config = ConfigDict(extra="ignore")


class LagCorrelationRequest(BaseModel):
    """双序列时滞关联扫描请求。"""

    driver_points: list[PointIn] = Field(
        min_length=1,
        description="驱动侧序列（时间戳格式与检测接口一致；无需预先排序，拒绝重复时间戳）",
    )
    response_points: list[PointIn] = Field(
        min_length=1,
        description="响应侧序列（要求同 driver_points）",
    )
    lag_seconds: list[FiniteFloat] = Field(
        min_length=1,
        max_length=10_000,
        description=(
            "候选时滞（秒，可为负）：对每个候选先把 driver 时间平移 +lag，"
            "再与 response 做一对一最近邻匹配"
        ),
    )
    match_tolerance_seconds: FiniteFloat = Field(
        ge=0,
        description="匹配容差（秒）：平移后两侧时间差不超过该值才允许配对",
    )
    min_pairs: int = Field(
        default=3,
        ge=2,
        le=1_000_000,
        description="计算 Pearson 相关所需的最少配对数（不足则该候选不可用）",
    )


class LagCandidateResult(BaseModel):
    """单个候选时滞的评估结果。"""

    lag_seconds: float = Field(description="候选时滞（秒）")
    pair_count: int = Field(description="一对一最近邻匹配得到的配对数")
    correlation: float | None = Field(
        description="配对序列的 Pearson 相关系数；候选不可用时为 null"
    )
    usable: bool = Field(description="该候选是否可用（配对数达标且两侧方差非零）")
    reason: str | None = Field(description="不可用原因；可用时为 null")


class LagCorrelationResponse(BaseModel):
    driver_point_count: int
    response_point_count: int
    match_tolerance_seconds: float
    min_pairs: int
    candidate_count: int = Field(description="候选时滞总数")
    usable_candidate_count: int = Field(description="可用候选数")
    candidates: list[LagCandidateResult] = Field(
        description="各候选时滞的评估结果（按请求顺序返回）"
    )
    best_lag: float | None = Field(
        description=(
            "最优时滞（秒）：可用候选中按 |correlation| 最大、配对数更多、"
            "|lag| 更小、lag 更小依次确定；无可用候选时为 null"
        )
    )

    model_config = ConfigDict(extra="ignore")


class Baseline(BaseModel):
    median: float
    mad: float
    robust_sd: float
    dispersion_method: Literal["mad", "std_fallback", "constant"] = Field(
        description="离散度来源：mad=1.4826·MAD；std_fallback=MAD 为 0 时回退样本标准差；constant=基线恒定"
    )
    scale: float
    window_size: int
    window_start_time: str
    window_end_time: str


class SegmentScore(BaseModel):
    max_abs_zscore: float
    max_signed_zscore: float
    max_zscore_index: int
    max_zscore_time: str
    mean_abs_zscore: float


class Trigger(BaseModel):
    index: int
    timestamp: str
    value: float
    zscore: float
    baseline_median: float
    baseline_mad: float
    dispersion_method: Literal["mad", "std_fallback", "constant"]
    deviation: float
    direction: Literal["上升", "下降"]
    reason: str


class AnomalySegment(BaseModel):
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    point_count: int
    anomaly_point_count: int
    baseline: Baseline
    score: SegmentScore
    triggers: list[Trigger]
    explanation: str
    sampling_segment_index: int = Field(
        default=0,
        description="该异常区间所属采样分段编号（从 0 开始）；未启用断档感知时恒为 0",
    )
    status: Literal["open", "acknowledged", "resolved"] = Field(
        default="open",
        description="处置状态：open → acknowledged → resolved 单向流转，新建检测的段恒为 open",
    )
    note: str | None = Field(default=None, description="处置备注；未填写时为 null")
    updated_at: str | None = Field(
        default=None, description="处置记录最近更新时间（UTC ISO-8601）"
    )


class SegmentStatusCounts(BaseModel):
    """一条检测的异常段处置状态计数。"""

    open: int = 0
    acknowledged: int = 0
    resolved: int = 0


class SegmentStatusUpdateRequest(BaseModel):
    """异常段处置状态提交。"""

    status: Literal["open", "acknowledged", "resolved"] = Field(
        description=(
            "目标处置状态。只允许 open → acknowledged → resolved 单向流转；"
            "提交与当前相同的状态为幂等重放（成功且不改变状态），其余跳转返回 409"
        )
    )
    note: str | None = Field(
        default=None,
        max_length=2000,
        description=(
            "可选处置备注（≤2000 字符）：合法流转时覆盖旧备注；"
            "幂等重放时仅在提供非 null 值时更新备注；省略则保留原备注"
        ),
    )


class SegmentStatusResponse(BaseModel):
    """异常段处置状态提交结果。"""

    detection_id: str
    segment_index: int
    status: Literal["open", "acknowledged", "resolved"]
    note: str | None
    updated_at: str = Field(description="处置记录最近更新时间（UTC ISO-8601）")
    outcome: Literal["transition", "idempotent"] = Field(
        description="transition=状态发生推进；idempotent=幂等重放（状态未变）"
    )
    status_counts: SegmentStatusCounts = Field(
        description="本次提交后该检测全部异常段的状态计数"
    )

    model_config = ConfigDict(extra="ignore")


class GapBoundary(BaseModel):
    """采样断档边界：断档发生在 before/after 两个相邻点之间。"""

    before_index: int = Field(description="断档前最后一个点的索引")
    after_index: int = Field(description="断档后第一个点的索引")
    before_time: str
    after_time: str
    gap_seconds: float = Field(description="断档两侧实际时间差（秒）")
    interval_seconds: float = Field(description="估算的常规采样间隔（相邻正时间差中位数，秒）")
    gap_ratio: float = Field(description="断档时间差 / 常规间隔的倍数")


class AnnotatedPoint(BaseModel):
    index: int
    timestamp: str
    value: float
    baseline_median: float
    baseline_mad: float
    baseline_robust_sd: float
    dispersion_method: Literal["mad", "std_fallback", "constant"]
    zscore: float
    degenerate_baseline: bool
    is_anomaly: bool
    baseline_generation: int = Field(
        default=0,
        description="评分该点时使用的基线代次；guarded 模式每次重建递增，rolling 恒为 0",
    )
    baseline_reset: bool = Field(
        default=False,
        description="该点是否触发了基线重建（用其所在异常段末尾 min_window 个点重建）",
    )
    sampling_segment_index: int = Field(
        default=0,
        description="该点所属采样分段编号（从 0 开始）；未启用断档感知时恒为 0",
    )


class DetectSummary(BaseModel):
    point_count: int
    window_size: int
    min_window: int
    threshold: float
    merge_gap: int
    baseline_policy: Literal["rolling", "guarded"] = "rolling"
    rebase_after: int | None = None
    baseline_rebuild_count: int = Field(default=0, description="guarded 模式基线重建（代次递增）次数")
    gap_reset_factor: float | None = Field(
        default=None, description="采样断档感知倍数；默认 null（关闭），结果与旧版完全兼容"
    )
    estimated_interval_seconds: float | None = Field(
        default=None, description="估算的常规采样间隔（相邻正时间差中位数，秒）；未启用断档感知时为 null"
    )
    gap_boundaries: list[GapBoundary] = Field(
        default_factory=list,
        description="检测到的采样断档边界（按时间顺序）；未启用断档感知时为空列表",
    )
    sampling_segment_count: int = Field(
        default=1, description="采样分段总数；未启用断档感知时恒为 1"
    )
    evaluated_start_index: int
    evaluated_point_count: int
    anomaly_point_count: int
    anomaly_segment_count: int


class DetectResponse(DetectSummary):
    id: str
    series_id: str | None = None
    created_at: str
    segments: list[AnomalySegment]
    points: list[AnnotatedPoint] | None = None
    status_counts: SegmentStatusCounts = Field(
        default_factory=SegmentStatusCounts,
        description="全部异常段的处置状态计数（open / acknowledged / resolved）",
    )

    model_config = ConfigDict(extra="ignore")


class DetectionListItem(BaseModel):
    id: str
    series_id: str | None
    created_at: str
    point_count: int
    anomaly_segment_count: int
    anomaly_point_count: int
    threshold: float
    window_size: int
    status_counts: SegmentStatusCounts = Field(
        default_factory=SegmentStatusCounts,
        description="该检测全部异常段的处置状态计数（open / acknowledged / resolved）",
    )


class DetectionListResponse(BaseModel):
    items: list[DetectionListItem]
    total: int


class IncidentAggregateRequest(BaseModel):
    """事件聚合请求：把多条已保存检测的异常段按时间邻近关系聚合。"""

    detection_ids: list[str] = Field(
        min_length=1,
        max_length=50,
        description="参与聚合的检测 ID（1~50 个，不可重复；任一不存在返回 404）",
    )
    gap_seconds: FiniteFloat = Field(
        ge=0,
        description=(
            "允许桥接的最大时间间隔（秒，非负）：两个异常段时间重叠、首尾相接或"
            "间隔不超过该值即归入同一事件；同一检测的相邻异常段同样按此规则合并"
        ),
    )

    @field_validator("detection_ids")
    @classmethod
    def _no_duplicate_ids(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("detection_ids 不得包含重复 ID")
        return v


class IncidentMember(BaseModel):
    """事件成员：一个检测中的一个异常段的稳定定位。"""

    detection_id: str = Field(description="来源检测 ID")
    series_id: str | None = Field(description="来源检测的业务序列标识；未设置时为 null")
    segment_index: int = Field(
        description="该异常段在来源检测结果 segments[] 中的序号（从 0 开始）"
    )
    start_index: int = Field(description="段在来源检测序列中的起始点索引")
    end_index: int = Field(description="段在来源检测序列中的结束点索引")
    start_time: str
    end_time: str
    point_count: int = Field(description="段覆盖的原始点数（含桥接正常点）")
    anomaly_point_count: int = Field(description="段内实际触发的异常点数")
    max_abs_zscore: float = Field(description="该段的最大绝对 z 分数")


class Incident(BaseModel):
    """聚合后的事件。"""

    start_time: str = Field(description="事件起始时间（最早成员段开始时间）")
    end_time: str = Field(description="事件结束时间（最晚成员段结束时间）")
    series_ids: list[str] = Field(
        description="覆盖的业务 series_id（去重、字典序排序；未设置 series_id 的检测不计入）"
    )
    detection_ids: list[str] = Field(description="覆盖的检测 ID（去重、字典序排序）")
    segment_count: int = Field(description="成员异常段总数")
    max_abs_zscore: float = Field(description="全部成员段中的最大绝对 z 分数")
    members: list[IncidentMember] = Field(
        description=(
            "成员段定位，按 (start_time, detection_id, segment_index) 稳定排序"
        )
    )


class IncidentAggregateResponse(BaseModel):
    gap_seconds: float
    detection_count: int = Field(description="参与聚合的检测数")
    segment_count: int = Field(description="参与聚合的异常段总数（无异常的检测贡献 0 段）")
    incident_count: int = Field(description="聚合后的事件数（所有检测均无异常时为 0）")
    incidents: list[Incident] = Field(description="事件列表，按事件起始时间稳定排序")

    model_config = ConfigDict(extra="ignore")


class HealthResponse(BaseModel):
    status: str
    database: str


class ResampleRequest(BaseModel):
    """无状态等间隔重采样请求。"""

    points: list[PointIn] = Field(
        min_length=1,
        description="原始采样点（至少 1 个）；无需预先排序，服务端自动排序并拒绝重复时间戳",
    )
    anchor_time: object = Field(
        description=(
            "网格对齐锚点：网格时刻为 anchor_time + k·interval_seconds（k 为整数）。"
            "时间戳格式与 points 一致（ISO-8601 或 epoch 秒/毫秒），可落在样本区间之外"
        )
    )
    interval_seconds: FiniteFloat = Field(
        gt=0,
        description=(
            "网格间隔（秒，>0 且有限），必须能表示为整数微秒，"
            "如 0.3 可以、0.3000005（半微秒）返回 422"
        ),
    )
    method: Literal["linear", "previous"] = Field(
        default="linear",
        description="补值方法：linear=前后样本线性插值；previous=取前一样本（前向填充）",
    )
    max_gap_seconds: FiniteFloat = Field(
        ge=0,
        description=(
            "允许补值的最大前后样本间隔（秒，≥0 且有限）：网格点前后样本间隔"
            "不超过该值时按 method 补值，否则返回 value=null、source=missing；"
            "不向首末样本之外外推"
        ),
    )

    @field_validator("anchor_time")
    @classmethod
    def _parseable_anchor(cls, v: object) -> object:
        try:
            parse_timestamp(v)
        except ValueError as exc:
            raise ValueError(f"anchor_time 无法解析: {exc}")
        return v

    @field_validator("interval_seconds")
    @classmethod
    def _whole_microsecond_interval(cls, v: float) -> float:
        from .resampling import whole_microseconds

        if whole_microseconds(v) is None:
            raise ValueError(
                f"interval_seconds={v} 不能表示为整数微秒（网格时间按整数微秒计算）"
            )
        return v


class ResampledPoint(BaseModel):
    """重采样网格上的单个点。"""

    index: int = Field(description="网格点序号（从 0 开始）")
    timestamp: str = Field(description="网格时刻（UTC ISO-8601）")
    value: float | None = Field(
        description="observed/interpolated 为数值；missing（断档）时为 null"
    )
    source: Literal["observed", "interpolated", "missing"] = Field(
        description="来源：observed=原始样本恰落该网格点；interpolated=按 method 补值；missing=前后样本间隔超过 max_gap_seconds"
    )


class MissingGapRange(BaseModel):
    """一段连续 missing 网格点及其包围样本（连续缺口范围）。"""

    start_index: int = Field(description="缺口首个网格点序号")
    end_index: int = Field(description="缺口末个网格点序号")
    start_time: str = Field(description="缺口首个网格点时刻")
    end_time: str = Field(description="缺口末个网格点时刻")
    point_count: int = Field(description="缺口内网格点数（含首尾）")
    before_time: str = Field(description="缺口前最后一个原始样本时刻")
    after_time: str = Field(description="缺口后第一个原始样本时刻")
    before_value: float = Field(description="缺口前最后一个原始样本值")
    after_value: float = Field(description="缺口后第一个原始样本值")
    sample_gap_seconds: float = Field(description="包围样本的实际时间差（秒）")
    max_gap_seconds: float = Field(description="请求的允许补值间隔（秒，回显）")


class ResampleResponse(BaseModel):
    anchor_time: str = Field(description="实际使用的网格锚点（UTC ISO-8601，回显）")
    interval_seconds: float
    method: Literal["linear", "previous"]
    max_gap_seconds: float
    range_start_time: str = Field(description="首样本时刻（网格闭区间下界）")
    range_end_time: str = Field(description="末样本时刻（网格闭区间上界）")
    grid_start_time: str | None = Field(description="首个网格点时刻；网格为空时为 null")
    grid_end_time: str | None = Field(description="末个网格点时刻；网格为空时为 null")
    input_point_count: int = Field(description="原始样本数（去重前请求点数经排序后）")
    point_count: int = Field(description="网格点总数（≤10000）")
    observed_count: int = Field(description="source=observed 的点数")
    interpolated_count: int = Field(description="source=interpolated 的点数")
    missing_count: int = Field(description="source=missing 的点数")
    points: list[ResampledPoint] = Field(description="按时间升序的网格点")
    missing_ranges: list[MissingGapRange] = Field(
        description="连续缺口范围列表（按时间升序）；无缺口时为空列表"
    )

    model_config = ConfigDict(extra="ignore")


class CounterRatesRequest(BaseModel):
    """累计计数器速率计算请求（无状态）。"""

    series_id: str | None = Field(
        default=None,
        max_length=128,
        description="可选的业务序列标识，随响应回显",
    )
    points: list[PointIn] = Field(
        min_length=2,
        description="带时间戳的累计读数（至少 2 个）；无需预先排序，服务端自动排序并拒绝重复时间戳",
    )
    max_counter: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "计数器满量程（≥0 且有限）：负增量且前值距 max_counter、后值距 0 "
            "均不超过 wrap_tolerance 时按回卷计算；省略时不做回卷判定，负增量一律标重置"
        ),
    )
    wrap_tolerance: FiniteFloat = Field(
        default=0.0,
        ge=0,
        description=(
            "回卷容差（≥0 且有限）：前值距 max_counter 与后值距 0 都不超过该值"
            "才判定为回卷；默认 0（要求恰好在边界上）。仅在提供 max_counter 时有意义"
        ),
    )
    max_gap_seconds: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "允许计入增量的最大相邻时间跨度（秒，≥0 且有限，含等号）：跨度超过该值"
            "的区间标 unknown（增量 / 速率为 null）并切断连续 epoch；省略时不做断档判定"
        ),
    )

    @model_validator(mode="after")
    def _check_wrap_params(self) -> "CounterRatesRequest":
        if self.max_counter is not None and self.wrap_tolerance > self.max_counter:
            raise ValueError(
                f"wrap_tolerance 不能大于 max_counter"
                f"（wrap_tolerance={self.wrap_tolerance} > max_counter={self.max_counter}）"
            )
        return self


class CounterRateInterval(BaseModel):
    """相邻两个读数之间的逐区间结果。"""

    index: int = Field(description="区间序号（从 0 开始，共 point_count − 1 个）")
    start_index: int = Field(description="区间起点读数在排序后序列中的索引")
    end_index: int = Field(description="区间终点读数在排序后序列中的索引")
    start_time: str
    end_time: str
    elapsed_seconds: float = Field(description="区间实际时间跨度（秒）")
    from_value: float = Field(description="区间起点读数")
    to_value: float = Field(description="区间终点读数")
    delta: float | None = Field(
        description="区间增量：normal=后值−前值，wrap=(max_counter−前值)+后值；"
        "reset / unknown 时为 null"
    )
    rate_per_second: float | None = Field(
        description="区间速率 = delta / elapsed_seconds；reset / unknown 时为 null"
    )
    reason: Literal["normal", "wrap", "reset", "unknown"] = Field(
        description=(
            "判定原因：normal=正常非负增量；wrap=按回卷计算的负增量；"
            "reset=负增量但不满足回卷条件（重置边界）；unknown=时间跨度超 max_gap_seconds"
        )
    )
    reason_detail: str = Field(description="判定原因的中文解释")
    epoch_index: int | None = Field(
        description="该区间所属连续 epoch 序号（从 0 开始）；reset / unknown 不属于任何 epoch，为 null"
    )


class CounterRateEpoch(BaseModel):
    """连续 epoch：被重置 / unknown 切断的一段连续已知增量。"""

    epoch_index: int = Field(description="epoch 序号（按时间顺序从 0 开始）")
    start_index: int = Field(description="epoch 首个读数在排序后序列中的索引")
    end_index: int = Field(description="epoch 末个读数在排序后序列中的索引")
    start_time: str
    end_time: str
    point_count: int = Field(description="epoch 内读数点数（= interval_count + 1）")
    interval_count: int = Field(description="epoch 内已知区间数（normal + wrap）")
    total_delta: float = Field(description="epoch 累计增量（各区间增量之和）")
    duration_seconds: float = Field(description="epoch 总时间跨度（首末读数之差，秒）")
    rate_per_second: float = Field(description="epoch 平均速率 = total_delta / duration_seconds")


class CounterRatesResponse(BaseModel):
    series_id: str | None = None
    point_count: int = Field(description="排序后的读数点数")
    interval_count: int = Field(description="区间总数（= point_count − 1）")
    max_counter: float | None = Field(description="回显满量程；未提供时为 null")
    wrap_tolerance: float = Field(description="回显回卷容差")
    max_gap_seconds: float | None = Field(description="回显断档上限；未提供时为 null")
    normal_count: int = Field(description="reason=normal 的区间数")
    wrap_count: int = Field(description="reason=wrap 的区间数")
    reset_count: int = Field(description="reason=reset 的区间数")
    unknown_count: int = Field(description="reason=unknown 的区间数")
    epoch_count: int = Field(description="连续 epoch 总数")
    total_delta: float = Field(
        description="汇总累计增量（仅 normal / wrap 区间，排除重置与 unknown）"
    )
    total_duration_seconds: float = Field(
        description="汇总时间跨度（仅 normal / wrap 区间的跨度之和，秒）"
    )
    rate_per_second: float | None = Field(
        description="汇总平均速率 = total_delta / total_duration_seconds；无已知区间时为 null"
    )
    intervals: list[CounterRateInterval] = Field(description="按时间升序的逐区间结果")
    epochs: list[CounterRateEpoch] = Field(
        description="连续 epoch 列表（按时间升序）；全部区间均为重置 / unknown 时为空列表"
    )

    model_config = ConfigDict(extra="ignore")
