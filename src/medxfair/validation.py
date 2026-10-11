import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from .statistics import safe_div

def metrics(y, p, flags):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); flags = np.asarray(flags, dtype=int)
    tp = int(((y==1)&(flags==1)).sum()); tn = int(((y==0)&(flags==0)).sum())
    fp = int(((y==0)&(flags==1)).sum()); fn = int(((y==1)&(flags==0)).sum())
    positives = tp+fn; negatives = tn+fp
    result = {"n":len(y),"positives":positives,"negatives":negatives,"tp":tp,"tn":tn,"fp":fp,"fn":fn,
       "prevalence":safe_div(positives,len(y)),"accuracy":safe_div(tp+tn,len(y)),
       "precision":safe_div(tp,tp+fp),"recall":safe_div(tp,positives),"sensitivity":safe_div(tp,positives),
       "specificity":safe_div(tn,negatives),"fnr":safe_div(fn,positives),"fpr":safe_div(fp,negatives),
       "f1":safe_div(2*tp,2*tp+fp+fn),"selection_rate":float(flags.mean()),
       "mean_probability":float(p.mean()),"calibration_gap":float(p.mean()-y.mean()),
       "brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1]))}
    result["auroc"] = float(roc_auc_score(y,p)) if positives and negatives else float("nan")
    result["auprc"] = float(average_precision_score(y,p)) if positives else float("nan")
    return result
