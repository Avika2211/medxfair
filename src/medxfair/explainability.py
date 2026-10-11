"""Optional SHAP-based descriptive attribution audit; not causal inference."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .exceptions import AuditError

@dataclass
class ExplanationResult:
    global_importance: pd.DataFrame
    subgroup_importance: pd.DataFrame
    metadata: dict

def explain_predictions(predict_fn, features: pd.DataFrame, *, groups=None,
                        background_size=40, sample_size=80, seed=42, max_evals=None):
    """Compute probability-scale SHAP attribution for a callable returning P(Y=1).

    Permutation SHAP with a background masker; only sampled rows are explained.
    Requires pip install medxfair[explain]. Features must be numeric and finite.
    No raw feature rows or per-patient explanations are returned.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install optional dependency: pip install medxfair[explain]") from exc
    if not isinstance(features,pd.DataFrame) or features.empty or not features.columns.is_unique:
        raise AuditError("features must be a nonempty DataFrame with unique column names")
    x=features.reset_index(drop=True)
    try: vals=x.to_numpy(dtype=float)
    except (TypeError,ValueError) as exc: raise AuditError("SHAP feature matrix must be numeric") from exc
    if not np.isfinite(vals).all(): raise AuditError("SHAP feature matrix must be finite")
    if not callable(predict_fn): raise AuditError("predict_fn must be callable")
    if background_size < 1 or sample_size < 1: raise AuditError("sample sizes must be positive")
    rng=np.random.default_rng(seed)
    bg_idx=np.sort(rng.choice(len(x),size=min(background_size,len(x)),replace=False))
    sm_idx=np.sort(rng.choice(len(x),size=min(sample_size,len(x)),replace=False))
    def wrapped(array):
        inp=pd.DataFrame(np.asarray(array),columns=x.columns)
        out=np.asarray(predict_fn(inp),dtype=float).reshape(-1)
        if len(out)!=len(inp) or not np.isfinite(out).all() or np.any((out<0)|(out>1)):
            raise AuditError("predict_fn must return finite probabilities in [0,1]")
        return out
    nfeatures=len(x.columns)
    budget=max_evals if max_evals is not None else max(2*nfeatures+1, 30)
    if budget < 2*nfeatures+1: raise AuditError("max_evals must be at least 2*n_features+1")
    explainer=shap.Explainer(wrapped,shap.maskers.Independent(x.iloc[bg_idx]),algorithm="permutation",seed=seed)
    values=np.asarray(explainer(x.iloc[sm_idx],max_evals=budget).values)
    if values.shape!=(len(sm_idx),nfeatures): raise AuditError("Unexpected SHAP output dimensions")
    importance=pd.DataFrame({"feature":x.columns,"mean_abs_shap":np.abs(values).mean(axis=0),"mean_shap":values.mean(axis=0)}).sort_values("mean_abs_shap",ascending=False).reset_index(drop=True)
    group_rows=[]
    if groups is not None:
        g=pd.Series(groups).reset_index(drop=True)
        if len(g)!=len(x): raise AuditError("groups must match feature row count")
        for label in sorted(g.iloc[sm_idx].fillna("<MISSING>").astype(str).unique()):
            mask=(g.iloc[sm_idx].fillna("<MISSING>").astype(str).to_numpy()==label)
            for j,col in enumerate(x.columns):
                group_rows.append({"group":label,"feature":col,"n_explained":int(mask.sum()),"mean_abs_shap":float(np.abs(values[mask,j]).mean()),"mean_shap":float(values[mask,j].mean())})
    return ExplanationResult(importance,pd.DataFrame(group_rows,columns=["group","feature","n_explained","mean_abs_shap","mean_shap"]),
        {"method":"permutation_SHAP","output_scale":"predicted_probability","background_n":len(bg_idx),"explained_n":len(sm_idx),"seed":seed,"warning":"Attributions depend on background, feature dependence, and model; not causal effects"})
