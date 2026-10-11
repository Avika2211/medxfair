from medxfair import MedxFair

def test_intersections(sample,config):
    table=MedxFair(config).run(sample).subgroup_metrics
    assert set(table.depth)=={1,2,3}

def test_missing_and_no_mutation(sample,config):
    d=sample.copy()
    d.loc[0,"payer"]=None
    before=d.copy(deep=True)
    report=MedxFair(config).run(d)
    assert report.subgroup_metrics.group.str.contains("<MISSING>",regex=False).any()
    assert d.equals(before)
