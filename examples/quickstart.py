from pathlib import Path
import pandas as pd
from medxfair import MedxFair,AuditConfig

root=Path(__file__).resolve().parent.parent
df=pd.read_csv(root/"examples"/"example_predictions.csv")
config=AuditConfig(outcome_col="readmitted_30d",probability_col="predicted_probability",protected_attributes=["race","payer","age_group"],min_group_size=5,min_positive_support=2,min_negative_support=2)
report=MedxFair(config).run(df)
report.export_csv(root/"outputs")
report.export_json(root/"outputs"/"audit.json")
report.export_html(root/"outputs"/"report.html")
print(report.overall_metrics)
