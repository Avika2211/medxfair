# Missing protected-attribute robustness

`audit_missing_group_bounds` calculates marginal FNR/FPR identification intervals under unrestricted assignment of missing group labels. Outcomes and predictions must be fully observed.

`audit_pairwise_disparity_bounds` computes coupled pairwise differences. An interval crossing zero means the direction is not identified by the permitted assignments. `max_missing_share` imposes a hypothetical within-confusion-cell cap per focal group; values below 1 require external justification and use fractional relaxed bounds.

These are **not sampling confidence intervals**. Wide intervals under substantial missingness are expected. See package functions for documented arguments.

```python
from medxfair import audit_missing_group_bounds, audit_pairwise_disparity_bounds
m = audit_missing_group_bounds(df, group_col="payer", outcome_col="y", prediction_col="flagged")
j = audit_pairwise_disparity_bounds(df, group_col="payer", outcome_col="y", prediction_col="flagged", pairs=[("MC","MD")])
```
