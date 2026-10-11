import numpy as np
import pandas as pd
from .exceptions import AuditError

MISSING = "<MISSING>"

def normalized_groups(df, attributes):
    out = df[list(attributes)].copy()
    for col in attributes:
        s = out[col].astype("string")
        s = s.mask((s == MISSING).fillna(False), "<OBSERVED:MISSING>")
        out[col] = s.fillna(MISSING)
    return out

def validate(df, config):
    if not isinstance(df, pd.DataFrame) or df.empty: raise AuditError("expected a nonempty pandas DataFrame")
    required = [config.outcome_col, config.probability_col, *config.protected_attributes]
    required += [x for x in (config.patient_id_col, config.record_id_col) if x]
    if len(set(required)) != len(required): raise AuditError("configuration columns must be distinct")
    absent = [x for x in required if x not in df.columns]
    if absent: raise AuditError(f"missing columns: {absent}")
    y = pd.to_numeric(df[config.outcome_col], errors="coerce")
    if y.isna().any() or not y.isin([0,1]).all(): raise AuditError("outcome must be nonmissing binary 0/1")
    p = pd.to_numeric(df[config.probability_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise AuditError("probabilities must be finite numeric values in [0,1]")
    return y.to_numpy(dtype=int), p

def predicted_flags(probabilities, config):
    n = len(probabilities)
    if config.evaluation_mode == "threshold":
        return (probabilities >= config.threshold).astype(int), {"threshold":config.threshold, "selected_count":int((probabilities >= config.threshold).sum())}
    k = int(np.ceil(n * config.review_budget))
    order = np.lexsort((np.arange(n), -probabilities))
    flags = np.zeros(n, dtype=int)
    flags[order[:k]] = 1
    return flags, {"review_budget":config.review_budget,"selected_count":k,"cutoff_probability":float(probabilities[order[k-1]])}
