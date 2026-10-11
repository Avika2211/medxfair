import pandas as pd
from .utils import normalized_groups

def cohort(df, y, config):
    d = {"records":len(df),"positive_outcomes":int(y.sum()),"prevalence":float(y.mean())}
    if config.patient_id_col:
        d["unique_patients"] = int(df[config.patient_id_col].nunique(dropna=True))
        d["repeated_patient_records"] = int(df[config.patient_id_col].duplicated(keep=False).sum())
    if config.record_id_col:
        d["duplicate_record_ids"] = int(df[config.record_id_col].duplicated(keep=False).sum())
    return d

def missingness(df):
    return pd.DataFrame([{"column":c,"missing_count":int(df[c].isna().sum()),"missing_rate":float(df[c].isna().mean()),"dtype":str(df[c].dtype)} for c in df.columns])

def missingness_by_groups(df, config):
    rows=[]; attrs=config.protected_attributes
    groups=normalized_groups(df,attrs)
    for target in attrs:
        for by in [x for x in attrs if x!=target]+[config.outcome_col]:
            keys=groups[by] if by in groups else df[by].astype("string")
            for label, idx in keys.groupby(keys,sort=True,dropna=False).groups.items():
                subset=df.loc[idx,target]
                rows.append({"target":target,"by":by,"group":str(label),"n":len(subset),"missing_rate":float(subset.isna().mean())})
    return pd.DataFrame(rows,columns=["target","by","group","n","missing_rate"])
