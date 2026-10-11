"""Missing protected-group labels: partial-identification error-rate bounds."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class MissingGroupBounds:
    table: pd.DataFrame
    metadata: dict
    def export_csv(self, path): self.table.to_csv(path,index=False)

def audit_missing_group_bounds(data, *, group_col, outcome_col, prediction_col, threshold=0.5, groups=None, missing_values=None):
    for col in (group_col,outcome_col,prediction_col):
        if col not in data: raise ValueError(f"Missing column: {col}")
    if data.empty or not 0 <= threshold <= 1: raise ValueError("Invalid data/threshold")
    y=pd.to_numeric(data[outcome_col],errors="coerce").to_numpy(dtype=float)
    p=pd.to_numeric(data[prediction_col],errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isin(y,[0,1]).all(): raise ValueError("Invalid outcome")
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("Invalid predictions")
    labels=data[group_col].astype("string")
    missing=labels.isna()
    if missing_values is not None: missing=missing|labels.isin([str(v) for v in missing_values]).fillna(False)
    missing=missing.to_numpy(dtype=bool)
    observed=labels[~missing]
    names=list(observed.unique()) if groups is None else [str(g) for g in groups]
    if len(names)<2 or len(set(names))!=len(names): raise ValueError("At least two unique groups required")
    if not observed.isin(names).all(): raise ValueError("groups must contain observed categories")
    pred=p>=threshold
    cells=dict(fn=(y==1)&~pred,tp=(y==1)&pred,fp=(y==0)&pred,tn=(y==0)&~pred)
    missing_counts={k:int((v&missing).sum()) for k,v in cells.items()}
    def ratio(a,b): return float(a/b) if b else float("nan")
    rows=[]
    for g in names:
        m=labels.eq(g).fillna(False).to_numpy(dtype=bool)&~missing
        a,b,c,d=[int((m&cells[k]).sum()) for k in ("fn","tp","fp","tn")]
        u,v,w,z=[missing_counts[k] for k in ("fn","tp","fp","tn")]
        rows.append(dict(group=g,observed_n=int(m.sum()),observed_positives=a+b,observed_negatives=c+d,observed_fn=a,observed_tp=b,observed_fp=c,observed_tn=d,observed_fnr=ratio(a,a+b),fnr_lower=ratio(a,a+b+v),fnr_upper=ratio(a+u,a+b+u),observed_fpr=ratio(c,c+d),fpr_lower=ratio(c,c+d+z),fpr_upper=ratio(c+w,c+d+w),fnr_identified=bool(u==0 and v==0 and a+b>0),fpr_identified=bool(w==0 and z==0 and c+d>0)))
    return MissingGroupBounds(pd.DataFrame(rows),dict(n_records=len(data),missing_group_n=int(missing.sum()),missing_group_fraction=float(missing.mean()),missing_confusion_counts=missing_counts,threshold=float(threshold),assumption="Unrestricted missing-group assignments",interpretation="Marginal identification bounds, not sampling confidence intervals",joint_attainability=False,outcome_and_predictions_fully_observed=True))

def mask_observed_groups(data, *, group_col, outcome_col, prediction_col, fraction, mechanism="MCAR",random_state=42):
    if not 0<=fraction<1: raise ValueError("Invalid masking fraction")
    if mechanism not in {"MCAR","OUTCOME_DEPENDENT","ERROR_DEPENDENT"}: raise ValueError("Invalid masking mechanism")
    result=data.copy(deep=True)
    observed=np.flatnonzero(result[group_col].notna().to_numpy())
    n=int(round(len(observed)*fraction))
    if not n:return result
    rng=np.random.default_rng(random_state)
    y=result[outcome_col].to_numpy()[observed];p=result[prediction_col].to_numpy()[observed]
    weight=np.ones(len(observed))
    if mechanism=="OUTCOME_DEPENDENT":weight=np.where(y==1,5.,1.)
    if mechanism=="ERROR_DEPENDENT":weight=np.where((y==1)&(p<.5),7.,1.)
    selected=rng.choice(observed,size=n,replace=False,p=weight/weight.sum())
    result.iloc[selected,result.columns.get_loc(group_col)]=np.nan
    return result
