"""MedxFair: retrospective fairness auditing and optional diagnostic extensions."""
from .config import AuditConfig
from .core import MedxFair
from .reporting import AuditReport
from .explainability import explain_predictions, ExplanationResult
from .sensitivity import feature_substitution, SensitivityResult
from .mitigation import review_candidates, ReviewConstraints, MitigationReview
__version__ = "0.4.0"
__all__ = ["AuditConfig","MedxFair","AuditReport","explain_predictions","ExplanationResult","feature_substitution","SensitivityResult","review_candidates","ReviewConstraints","MitigationReview"]

from .missingness_robust import audit_missing_group_bounds, mask_observed_groups, MissingGroupBounds
from .disparity_robust import audit_pairwise_disparity_bounds, DisparityBounds
