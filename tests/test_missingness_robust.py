import pandas as pd
from medxfair import audit_missing_group_bounds,mask_observed_groups

def test_missing_labels():
    df=pd.DataFrame(dict(g=["A","B",pd.NA,pd.NA],y=[1,1,1,1],p=[0,1,0,1]))
    t=audit_missing_group_bounds(df,group_col="g",outcome_col="y",prediction_col="p").table
    assert len(t)==2
    assert t.iloc[0].fnr_lower <= t.iloc[0].observed_fnr <= t.iloc[0].fnr_upper

def test_mask_preserves_input():
    df=pd.DataFrame(dict(g=["A","B","A","B"],y=[1,0,1,0],p=[0.,1.,0.,1.]))
    m=mask_observed_groups(df,group_col="g",outcome_col="y",prediction_col="p",fraction=.5)
    assert m.g.isna().sum()==2
    assert df.g.notna().all()
