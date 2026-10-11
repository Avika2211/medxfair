# MedxFair-ROBUST (v0.4.0)

Research-oriented Python package for retrospective fairness auditing of binary prediction models, including intersectional subgroup analysis, descriptive SHAP explanations, input substitution sensitivity checks, model mitigation review, and partial-identification bounds under missing protected attributes.

**Research use only:** model disparities do not prove discrimination; feature substitutions are not causal interventions; mitigation eligibility is not clinical safety. Missing-membership identification bounds are not sampling confidence intervals.

## Install

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/quickstart.py
python -m build
```

## Quickstart

```python
import pandas as pd
from medxfair import MedxFair, AuditConfig
df = pd.read_csv("examples/example_predictions.csv")
config = AuditConfig(outcome_col="readmitted_30d", probability_col="predicted_probability", protected_attributes=["race","payer","age_group"], min_group_size=5,min_positive_support=2,min_negative_support=2)
report = MedxFair(config).run(df)
report.export_json("outputs/audit.json")
report.export_csv("outputs")
report.export_html("outputs/report.html")
```

The package accepts precomputed binary-outcome probabilities and does not retrain the model. Threshold mode uses p>=threshold; budget mode selects exactly ceil(budget*N) high-risk rows, ties broken by input order. Missing attributes remain explicit categories; supported-group thresholds guard headline worst-group estimates. Wilson confidence intervals are pointwise, not simultaneous.

For partial-identification APIs, see `ROBUST.md`; for explanation, sensitivity and candidate review modules see `MILESTONE2.md`. The examples are synthetic software tests, not clinical data. Aggregate outputs can still disclose small-group information. Review before sharing.
