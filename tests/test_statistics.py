import math
import pytest
from medxfair.statistics import wilson_interval

def test_wilson():
    assert all(math.isnan(x) for x in wilson_interval(0,0))
    low,high=wilson_interval(5,10)
    assert low==pytest.approx(.236593,abs=1e-5)
    assert high==pytest.approx(.763407,abs=1e-5)
