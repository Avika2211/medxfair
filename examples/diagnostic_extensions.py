"""Synthetic demonstration of optional MedxFair diagnostics."""
import numpy as np
import pandas as pd
from medxfair import AuditConfig,feature_substitution,review_candidates
rng=np.random.default_rng(42)
features=pd.DataFrame({"age_proxy":rng.uniform(0,1,120),"comorbidity_proxy":rng.uniform(0,1,120)})
def predict(f):
    return np.clip(0.08+0.2*f.age_proxy.to_numpy()+0.15*f.comorbidity_proxy.to_numpy(),0,1)
p=predict(features);y=rng.binomial(1,p)
group=np.where(features.age_proxy>.5,"older","younger")
data=pd.DataFrame({"y":y,"p":p,"group":group})
config=AuditConfig("y","p",["group"],threshold=.2,min_group_size=10,min_positive_support=3,min_negative_support=3)
print(feature_substitution(predict,features,{"age_proxy":.5},groups=group).summary.to_string(index=False))
print(review_candidates(data,config,{"risk_shift":np.clip(p+.03,0,1)}).comparison.to_string(index=False))
