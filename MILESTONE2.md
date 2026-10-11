# MedxFair diagnostic extensions

`explain_predictions`: optional permutation SHAP; requires `pip install medxfair[explain]`. Provides descriptive model attributions on sampled numeric inputs, **not causal effects**.

`feature_substitution`: changes selected model-input values and measures shifts in predicted probabilities; not do-operator causal identification or recommended clinical intervention.

`review_candidates`: compares user-provided candidate probability vectors against baseline on an independent selection partition under configurable point-estimate constraints. `ELIGIBLE` is not clinical approval; do not select using final test results.
