from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class DisparityBounds:
    table: pd.DataFrame
    metadata: dict
    def export_csv(self, path): self.table.to_csv(path, index=False)

def audit_pairwise_disparity_bounds(data, *, group_col, outcome_col, prediction_col, pairs=None, threshold=0.5, max_missing_share=1.0, missing_values=None, tolerance=0.0):
    """Bounds for FNR_A-FNR_B and FPR_A-FPR_B with coupled unknown labels."""
    if not 0 <= max_missing_share <= 1 or not 0 <= threshold <= 1 or tolerance < 0:
        raise ValueError("Invalid threshold, allocation cap, or tolerance")
    for col in (group_col,outcome_col,prediction_col):
        if col not in data: raise ValueError(f"Missing column: {col}")
    if data.empty: raise ValueError("Empty data")
    y=pd.to_numeric(data[outcome_col],errors="coerce").to_numpy(dtype=float)
    p=pd.to_numeric(data[prediction_col],errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isin(y,[0,1]).all(): raise ValueError("Invalid outcomes")
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("Invalid probabilities")
    labels=data[group_col].astype("string")
    missing=labels.isna()
    if missing_values is not None: missing=missing|labels.isin([str(v) for v in missing_values]).fillna(False)
    missing=missing.to_numpy(dtype=bool)
    names=list(labels[~missing].dropna().unique())
    if pairs is None: pairs=[(names[i],names[j]) for i in range(len(names)) for j in range(i+1,len(names))]
    pred=p>=threshold
    cells={"fn":(y==1)&~pred,"tp":(y==1)&pred,"fp":(y==0)&pred,"tn":(y==0)&~pred}
    missing_counts={k:int((v&missing).sum()) for k,v in cells.items()}
    counts={}
    for g in names:
        membership=labels.eq(g).fillna(False).to_numpy(dtype=bool)&~missing
        counts[g]={k:int((membership&v).sum()) for k,v in cells.items()}
    def ratio(a,b): return float(a/b) if b>0 else float("nan")
    def bounds(a,b,metric):
        num,den=("fn","tp") if metric=="fnr" else ("fp","tn")
        an,ad=a[num],a[den];bn,bd=b[num],b[den]
        mn=max_missing_share*missing_counts[num];md=max_missing_share*missing_counts[den]
        return ratio(an,an+ad+md)-ratio(bn+mn,bn+bd+mn),ratio(an+mn,an+ad+mn)-ratio(bn,bn+bd+md),ratio(an,an+ad)-ratio(bn,bn+bd)
    rows=[]
    for a,b in pairs:
        a,b=str(a),str(b)
        if a==b or a not in counts or b not in counts: raise ValueError(f"Unknown pair {a}, {b}")
        for metric in ("fnr","fpr"):
            low,high,observed=bounds(counts[a],counts[b],metric)
            direction="UNDEFINED" if not np.isfinite(low+high) else "A_HIGHER" if low>tolerance else "B_HIGHER" if high < -tolerance else "NOT_IDENTIFIED"
            rows.append(dict(group_a=a,group_b=b,metric=metric,observed_difference=observed,difference_lower=low,difference_upper=high,direction=direction,group_a_observed_n=sum(counts[a].values()),group_b_observed_n=sum(counts[b].values()),max_missing_share=float(max_missing_share)))
    return DisparityBounds(pd.DataFrame(rows),dict(n=len(data),missing_group_n=int(missing.sum()),missing_confusion_counts=missing_counts,threshold=threshold,max_missing_share=max_missing_share,tolerance=tolerance,assumptions="Observed labels fixed; unknown labels allocated to A, B or other",interpretation="Coupled partial-identification bounds; not confidence intervals",caveat="Caps below 1 are fractional relaxations and require justification"))
