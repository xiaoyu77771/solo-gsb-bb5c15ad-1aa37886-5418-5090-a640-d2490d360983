"""异常段处置工作流：open → acknowledged → resolved 单向状态机。

每条已保存检测的每个异常段都持久化一条处置记录
（status / note / updated_at，默认 open，随检测创建时写入）。
状态只允许沿 open → acknowledged → resolved 前进：

- 提交与当前状态相同的目标状态为**幂等重放**（成功，状态不变；
  仅当请求附带非空 note 时更新 note 与 updated_at）；
- 合法前进一步（open→acknowledged、acknowledged→resolved）正常流转；
- 其余跳转（open→resolved 越级、acknowledged→open / resolved→open 回退、
  resolved→acknowledged 回退）均为非法，返回 409。

本模块只做纯函数判定，不触碰存储层。
"""

from __future__ import annotations

STATUS_OPEN = "open"
STATUS_ACKNOWLEDGED = "acknowledged"
STATUS_RESOLVED = "resolved"

#: 全部合法状态（请求校验用）
SEGMENT_STATUSES: tuple[str, ...] = (STATUS_OPEN, STATUS_ACKNOWLEDGED, STATUS_RESOLVED)

#: 允许的单步前进转移
_ALLOWED_FORWARD: dict[str, str] = {
    STATUS_OPEN: STATUS_ACKNOWLEDGED,
    STATUS_ACKNOWLEDGED: STATUS_RESOLVED,
}

#: 状态机判定结果
OUTCOME_IDEMPOTENT = "idempotent"  # 幂等重放：目标 == 当前
OUTCOME_TRANSITION = "transition"  # 合法前进一步
OUTCOME_CONFLICT = "conflict"  # 非法跳转


def decide(current: str, target: str) -> str:
    """判定从 current 提交 target 的处置结果（见模块 docstring）。"""
    if target == current:
        return OUTCOME_IDEMPOTENT
    if _ALLOWED_FORWARD.get(current) == target:
        return OUTCOME_TRANSITION
    return OUTCOME_CONFLICT
