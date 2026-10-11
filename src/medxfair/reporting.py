from dataclasses import dataclass
from pathlib import Path
from html import escape
import json
import math
import numpy as np
import pandas as pd

def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,(float,int)) and not isinstance(value,bool):
        return None if not math.isfinite(value) else value
    if pd.isna(value): return None
    return value

@dataclass
class AuditReport:
    metadata: dict
    cohort_summary: dict
    missingness: pd.DataFrame
    missingness_by_group: pd.DataFrame
    overall_metrics: dict
    subgroup_metrics: pd.DataFrame
    disparity_summary: pd.DataFrame
    warnings: list[dict]

    def as_dict(self):
        return clean({"metadata":self.metadata,"cohort_summary":self.cohort_summary,
            "overall_metrics":self.overall_metrics,"missingness":self.missingness.to_dict("records"),
            "missingness_by_group":self.missingness_by_group.to_dict("records"),
            "subgroup_metrics":self.subgroup_metrics.to_dict("records"),
            "disparity_summary":self.disparity_summary.to_dict("records"),"warnings":self.warnings})

    def export_json(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(self.as_dict(),indent=2,allow_nan=False),encoding="utf-8")

    def export_csv(self,directory):
        d=Path(directory);d.mkdir(parents=True,exist_ok=True)
        tables={"overall_metrics":pd.DataFrame([self.overall_metrics]),"subgroup_metrics":self.subgroup_metrics,
            "missingness":self.missingness,"missingness_by_group":self.missingness_by_group,
            "disparities":self.disparity_summary,"warnings":pd.DataFrame(self.warnings,columns=["code","severity","affected","message"])}
        for name,frame in tables.items():frame.to_csv(d/f"{name}.csv",index=False)

    def export_html(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        sections=[("Cohort",pd.DataFrame([self.cohort_summary])),("Overall metrics",pd.DataFrame([self.overall_metrics])),
          ("Missingness",self.missingness),("Missingness by group",self.missingness_by_group),
          ("Intersectional and marginal fairness",self.subgroup_metrics),("Disparity summary",self.disparity_summary),
          ("Warnings",pd.DataFrame(self.warnings))]
        html=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MedxFair Audit</title>',
        '<style>body{font:15px system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 18px;color:#17212b;background:#fafbfc}h1{color:#12364a}section{margin:28px 0}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #ddd;padding:7px;text-align:left}th{background:#e8eff3;position:sticky;top:0}.scroll{overflow:auto;max-height:470px;background:white;border:1px solid #ddd}p{line-height:1.5}</style></head><body><h1>MedxFair — Audit Report</h1>',
        '<p>Retrospective prediction audit. Differences do not establish discrimination or causality. This is not a clinical safety assessment. Sparse-group estimates may be unstable; intervals are unadjusted for multiplicity.</p>']
        for title,frame in sections:
            html.extend([f'<section><h2>{escape(title)}</h2><div class="scroll">',frame.to_html(index=False,escape=True,na_rep='—'),'</div></section>'])
        html.append('</body></html>');path.write_text(''.join(html),encoding='utf-8')
