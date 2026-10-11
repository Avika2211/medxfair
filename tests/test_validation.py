import numpy as np
import pytest
from sklearn.metrics import roc_auc_score,average_precision_score
from medxfair import MedxFair,AuditConfig
from medxfair.utils import predicted_flags
from medxfair.exceptions import AuditError

def test_confusion_and_auc(sample,config):
    r=MedxFair(config).run(sample).overall_metrics
    assert (r["tp"],r["tn"],r["fp"],r["fn"])==(2,2,2,2)
    assert r["fnr"]==.5 and r["fpr"]==.5
    assert r["auroc"]==pytest.approx(roc_auc_score(sample.y,sample.p))
    assert r["auprc"]==pytest.approx(average_precision_score(sample.y,sample.p))

def test_one_class(sample,config):
    d=sample.copy();d["y"]=1
    r=MedxFair(config).run(d)
    assert np.isnan(r.overall_metrics["auroc"])
    assert any(w["code"]=="SINGLE_CLASS" for w in r.warnings)

def test_no_predicted_positives(sample,config):
    d=sample.copy();d["p"]=0
    r=MedxFair(config).run(d)
    assert np.isnan(r.overall_metrics["precision"])

def test_invalid_probabilities(sample,config):
    for value in [1.2,-.1,float("nan"),float("inf"),"oops"]:
        d=sample.copy();d.loc[0,"p"]=value
        with pytest.raises(AuditError):MedxFair(config).run(d)

def test_budget_ties():
    c=AuditConfig("y","p",["race"],evaluation_mode="budget",review_budget=.25)
    flags,_=predicted_flags(np.array([.5,.5,.5,.5,.1,.1,.1,.1]),c)
    assert flags.tolist()==[1,1,0,0,0,0,0,0]
