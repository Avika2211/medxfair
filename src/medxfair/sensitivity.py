"""Non-causal feature-substitution sensitivity diagnostics."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .exceptions import AuditError

@dataclass
class SensitivityResult:
    summary: pd.DataFrame
    by_group: pd.DataFrame
    metadata: dict

def feature_substitution(predict_fn, features:pd.DataFrame, interventions:dict, *, groups=None):
    """Compare model probabilities after hypothetical feature substitutions.

    Changes to feature values are not do-operator interventions, counterfactual
    outcomes, or evidence of a causal pathway. Never modify the input dataframe.
    """
    if not isinstance(features,pd.DataFrame) or features.empty or not callable(predict_fn):
        raise AuditError("Provide nonempty features DataFrame and callable predict_fn")
    if not interventions: raise AuditError("interventions cannot be empty")
    x=features.reset_index(drop=True).copy(deep=True)
    def predict(frame):
        p=np.asarray(predict_fn(frame),dtype=float).reshape(-1)
        if len(p)!=len(frame) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise AuditError("predict_fn must return finite probabilities in [0,1] for every row")
        return p
    base=predict(x)
    g=None
    if groups is not None:
        g=pd.Series(groups).reset_index(drop=True).fillna("<MISSING>").astype(str)
        if len(g)!=len(x):raise AuditError("groups length must match features")
    summary=[];by_group=[]
    for col,value in interventions.items():
        if col not in x: raise AuditError(f"Unknown intervention feature: {col}")
        changed=x.copy(deep=True)
        if callable(value):raise AuditError("Callable interventions prohibited; provide a fixed scalar value")
        if not np.isscalar(value):raise AuditError("Intervention values must be scalars")
        changed[col]=value
        delta=predict(changed)-base
        summary.append({"feature":col,"substituted_value":str(value),"n":len(delta),"mean_probability_change":float(delta.mean()),"mean_abs_change":float(np.abs(delta).mean()),"min_change":float(delta.min()),"max_change":float(delta.max())})
        if g is not None:
            for label in sorted(g.unique()):
                mask=(g.to_numpy()==label)
                by_group.append({"feature":col,"group":label,"n":int(mask.sum()),"mean_probability_change":float(delta[mask].mean()),"mean_abs_change":float(np.abs(delta[mask]).mean())})
    return SensitivityResult(pd.DataFrame(summary),pd.DataFrame(by_group,columns=["feature","group","n","mean_probability_change","mean_abs_change"]),
        {"method":"feature_substitution","warning":"Associational model sensitivity only; not causal identification or a clinical intervention"})
