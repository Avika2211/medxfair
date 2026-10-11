from itertools import combinations
import numpy as np
import pandas as pd
from .statistics import wilson_interval
from .validation import metrics
from .utils import normalized_groups

PROPORTIONS = {"fnr":("fn","positives"),"fpr":("fp","negatives"),"sensitivity":("tp","positives"),"specificity":("tn","negatives"),"prevalence":("positives","n")}

def subgroups(df, y, p, flags, config, overall):
    groups=normalized_groups(df,config.protected_attributes)
    rows=[]; depth=config.max_intersection_depth or len(config.protected_attributes)
    for k in range(1,depth+1):
        for attrs in combinations(config.protected_attributes,k):
            for values, indices in groups.groupby(list(attrs),dropna=False,sort=True).indices.items():
                if k==1: values=(values,)
                ix=np.asarray(indices,dtype=int)
                m=metrics(y[ix],p[ix],flags[ix])
                eligible = m["n"] >= config.min_group_size and m["positives"] >= config.min_positive_support and m["negatives"] >= config.min_negative_support
                status = "UNDEFINED_METRIC" if m["positives"]==0 or m["negatives"]==0 else ("SUPPORTED" if eligible else "LOW_SUPPORT")
                row={"depth":k,"attributes":" × ".join(attrs),"group":" | ".join(f"{a}={v}" for a,v in zip(attrs,values)),"support_status":status,**m}
                for key,(num,den) in PROPORTIONS.items():
                    row[f"{key}_ci_low"],row[f"{key}_ci_high"]=wilson_interval(m[num],m[den],config.confidence_level)
                for key in ("fnr","fpr","selection_rate","calibration_gap"):
                    row[f"{key}_difference"] = m[key]-overall[key]
                rows.append(row)
    return pd.DataFrame(rows)

def disparities(table):
    rows=[]
    for depth, t in table.groupby("depth"):
        eligible=t[t.support_status=="SUPPORTED"]
        for metric in ("fnr","fpr","selection_rate","calibration_gap"):
            vals=eligible[metric].dropna()
            if len(vals):
                rows.append({"depth":int(depth),"metric":metric,"eligible_groups":len(vals),"worst_value":float(vals.max()),"between_group_gap":float(vals.max()-vals.min()),"worst_group":str(eligible.loc[vals.idxmax(),"group"])})
            else:
                rows.append({"depth":int(depth),"metric":metric,"eligible_groups":0,"worst_value":float("nan"),"between_group_gap":float("nan"),"worst_group":None})
    return pd.DataFrame(rows)
