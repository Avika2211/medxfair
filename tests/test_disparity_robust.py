import pandas as pd
import pytest
from medxfair import audit_pairwise_disparity_bounds

def test_pair_bounds():
    d=pd.DataFrame(dict(group=["A","A","B","B",pd.NA,pd.NA],y=[1,1,1,1,1,1],p=[0,1,0,1,0,1]))
    t=audit_pairwise_disparity_bounds(d,group_col="group",outcome_col="y",prediction_col="p",pairs=[("A","B")]).table
    f=t[t.metric=="fnr"].iloc[0]
    assert f.difference_lower==pytest.approx(-1/3)
    assert f.difference_upper==pytest.approx(1/3)
    assert f.direction=="NOT_IDENTIFIED"
