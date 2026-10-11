from dataclasses import dataclass, asdict
from typing import Mapping
import numpy as np
import pandas as pd
from .config import AuditConfig
from .core import MedxFair
from .exceptions import AuditError

@dataclass(frozen=True)
class ReviewConstraints:
    max_auroc_drop: float = 0.01
    max_brier_increase: float = 0.005
    max_fnr_increase: float = 0.03
    max_worst_group_fnr_increase: float = 0.03
    min_worst_group_fnr_improvement: float = 0.0
    def __post_init__(self):
        if any(v < 0 or not np.isfinite(v) for v in asdict(self).values()):
            raise AuditError('Constraints must be finite and nonnegative')

@dataclass
class MitigationReview:
    comparison: pd.DataFrame
    metadata: dict

def review_candidates(data: pd.DataFrame, config: AuditConfig, candidate_probabilities: Mapping[str, object], *, constraints: ReviewConstraints = ReviewConstraints()):
    if not isinstance(data, pd.DataFrame) or data.empty or not candidate_probabilities:
        raise AuditError('Nonempty data and candidate probabilities required')
    base = MedxFair(config).run(data)
    def worst(report):
        t = report.subgroup_metrics
        t = t[(t.support_status == 'SUPPORTED') & np.isfinite(t.fnr)]
        return float(t.fnr.max()) if len(t) else float('nan')
    bm = base.overall_metrics; bw = worst(base)
    rows = [dict(candidate='baseline', decision='BASELINE', reason='reference', auroc=bm['auroc'], brier=bm['brier'], fnr=bm['fnr'], worst_supported_fnr=bw, delta_auroc=0.,delta_brier=0.,delta_fnr=0.,delta_worst_supported_fnr=0.)]
    for name, prob in candidate_probabilities.items():
        if name == 'baseline': raise AuditError('baseline is reserved')
        p = np.asarray(prob, dtype=float).reshape(-1)
        if len(p) != len(data) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any(): raise AuditError('Invalid candidate')
        tmp = data.copy(); tmp[config.probability_col] = p
        mreport = MedxFair(config).run(tmp)
        m = mreport.overall_metrics; w = worst(mreport)
        delta = dict(auroc=m['auroc'] - bm['auroc'], brier=m['brier'] - bm['brier'], fnr=m['fnr'] - bm['fnr'], worst_supported_fnr=w-bw)
        if not all(np.isfinite(v) for v in delta.values()): decision, reason = 'DEFER', 'Undefined evidence'
        elif delta['auroc'] < -constraints.max_auroc_drop or delta['brier'] > constraints.max_brier_increase or delta['fnr'] > constraints.max_fnr_increase or delta['worst_supported_fnr'] > constraints.max_worst_group_fnr_increase: decision, reason = 'REJECT', 'Constraint violated'
        elif delta['worst_supported_fnr'] < -constraints.min_worst_group_fnr_improvement: decision, reason = 'ELIGIBLE', 'Requires independent confirmation'
        else: decision, reason = 'DEFER', 'No qualifying improvement'
        rows.append(dict(candidate=str(name),decision=decision,reason=reason,auroc=m['auroc'],brier=m['brier'],fnr=m['fnr'],worst_supported_fnr=w,**{'delta_'+k:v for k,v in delta.items()}))
    return MitigationReview(pd.DataFrame(rows), dict(constraints=asdict(constraints),evaluation_mode=config.evaluation_mode,selection_only=True,warning='Point-estimate rules are not clinical safety certificates'))
