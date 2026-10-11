from dataclasses import asdict
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import AuditConfig
from .data_audit import cohort, missingness, missingness_by_groups
from .intersectional import subgroups, disparities
from .reporting import AuditReport
from .utils import validate, predicted_flags
from .validation import metrics

class MedxFair:
    """Audit precomputed probabilities for a binary outcome; does not train a model."""
    def __init__(self, config: AuditConfig): self.config=config

    def run(self, df: pd.DataFrame) -> AuditReport:
        """Return an offline audit; does not modify input or retain patient-level rows."""
        c=self.config
        data=df.reset_index(drop=True).copy()
        y,p=validate(data,c)
        flags,mode_info=predicted_flags(p,c)
        overall=metrics(y,p,flags)
        group_table=subgroups(data,y,p,flags,c,overall)
        miss=missingness(data)
        warnings=[]
        def warn(code,severity,affected,message):warnings.append({"code":code,"severity":severity,"affected":affected,"message":message})
        if not (y.min()!=y.max()):warn("SINGLE_CLASS","warning",c.outcome_col,"AUROC undefined with only one observed class")
        if not np.isfinite(overall["precision"]):warn("UNDEFINED_PRECISION","warning","overall","No predicted positives")
        for _,row in miss.iterrows():
            if row.missing_rate >= c.high_missingness_threshold and row.missing_count:
                warn("HIGH_MISSINGNESS","warning",row.column,f"Missing rate {row.missing_rate:.1%}")
        for a in c.protected_attributes:
            if data[a].isna().any():warn("MISSING_PROTECTED","warning",a,"Missing values audited as explicit category")
        if c.patient_id_col:
            repeated=int(data[c.patient_id_col].duplicated(keep=False).sum())
            if repeated:warn("REPEATED_PATIENT","warning",c.patient_id_col,f"{repeated} records have repeated patient identifiers; record-level uncertainty assumes independence")
        if c.record_id_col and data[c.record_id_col].duplicated(keep=False).any():
            warn("DUPLICATE_RECORD_ID","warning",c.record_id_col,"Duplicate record identifiers found")
        for _,r in group_table.iterrows():
            if r.support_status!="SUPPORTED":warn("SPARSE_OR_UNDEFINED_GROUP","warning",r.group,f"{r.support_status}: n={r.n}, positives={r.positives}, negatives={r.negatives}")
        warn("NO_SPLIT_PROVENANCE","info","evaluation","Prediction-only input cannot establish patient-disjoint train/test provenance")
        warn("MULTIPLE_COMPARISONS","info","subgroups","Wilson intervals are pointwise, not multiplicity-adjusted")
        return AuditReport(metadata={"package_version":"0.4.0","generated_at":datetime.now(timezone.utc).isoformat(),"config":asdict(c),"evaluation_mode":c.evaluation_mode,**mode_info},
           cohort_summary=cohort(data,y,c),missingness=miss,missingness_by_group=missingness_by_groups(data,c),
           overall_metrics=overall,subgroup_metrics=group_table,disparity_summary=disparities(group_table),warnings=warnings)
