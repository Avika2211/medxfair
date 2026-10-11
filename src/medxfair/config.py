from dataclasses import dataclass
from typing import Literal
from .exceptions import AuditError

@dataclass(frozen=True)
class AuditConfig:
    outcome_col: str
    probability_col: str
    protected_attributes: tuple[str, ...] | list[str]
    threshold: float = 0.5
    evaluation_mode: Literal["threshold", "budget"] = "threshold"
    review_budget: float | None = None
    min_group_size: int = 30
    min_positive_support: int = 10
    min_negative_support: int = 10
    max_intersection_depth: int | None = None
    confidence_level: float = 0.95
    patient_id_col: str | None = None
    record_id_col: str | None = None
    high_missingness_threshold: float = 0.2

    def __post_init__(self):
        attrs = tuple(self.protected_attributes)
        if not attrs or len(set(attrs)) != len(attrs):
            raise AuditError("protected_attributes must be a nonempty set of unique column names")
        object.__setattr__(self, "protected_attributes", attrs)
        if not (0 <= self.threshold <= 1): raise AuditError("threshold must be in [0,1]")
        if self.evaluation_mode not in ("threshold", "budget"): raise AuditError("invalid evaluation_mode")
        if self.evaluation_mode == "budget" and (self.review_budget is None or not 0 < self.review_budget <= 1):
            raise AuditError("budget mode requires review_budget in (0,1]")
        if self.min_group_size < 1 or self.min_positive_support < 1 or self.min_negative_support < 1:
            raise AuditError("support thresholds must be positive")
        if not 0 < self.confidence_level < 1: raise AuditError("confidence_level must be in (0,1)")
        if self.max_intersection_depth is not None and not 1 <= self.max_intersection_depth <= len(attrs):
            raise AuditError("invalid max_intersection_depth")
        if not 0 <= self.high_missingness_threshold <= 1: raise AuditError("invalid high_missingness_threshold")
