import json
from medxfair import MedxFair

def test_exports(sample,config,tmp_path):
    d=sample.copy();d["y"]=1
    report=MedxFair(config).run(d)
    report.export_json(tmp_path/"report.json")
    report.export_csv(tmp_path/"csv")
    report.export_html(tmp_path/"report.html")
    assert json.loads((tmp_path/"report.json").read_text())["overall_metrics"]["auroc"] is None
    assert (tmp_path/"csv"/"subgroup_metrics.csv").exists()
