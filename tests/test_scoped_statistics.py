import numpy as np
from scipy.stats import binom
from sevc.evaluation.scoped_statistics import exact_interval, paired_bootstrap


def test_zero_event_exact_bound_has_finite_sample_coverage():
    alpha=.05/6
    lo,hi=exact_interval(0,120,alpha)
    assert lo==0 and abs(binom.cdf(0,120,hi)-alpha/2)<1e-12
    assert exact_interval(0,0,alpha)==[None,None]


def test_paired_resampling_respects_units_and_fixed_seed():
    a=paired_bootstrap([1.,2.,3.],2026119000)
    b=paired_bootstrap([1.,2.,3.],2026119000)
    assert a==b and a['n']==3 and a['mean']==2
    assert a['interval'][0]>=1 and a['interval'][1]<=3
    assert paired_bootstrap([],1)['mean'] is None
