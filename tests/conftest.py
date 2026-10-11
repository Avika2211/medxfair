import pytest
import pandas as pd
from medxfair import AuditConfig

@pytest.fixture
def sample():
    return pd.DataFrame({"y":[1,1,0,0,1,0,1,0],"p":[.9,.2,.8,.1,.7,.3,.4,.6],"race":["A","A","A","A","B","B","B","B"],"payer":["X","Y","X","Y","X","Y","X","Y"],"age":["young","old","young","old","young","old","young","old"],"patient":[1,2,3,4,5,6,7,8]})

@pytest.fixture
def config():
    return AuditConfig(outcome_col="y",probability_col="p",protected_attributes=["race","payer","age"],min_group_size=2,min_positive_support=1,min_negative_support=1,patient_id_col="patient")
