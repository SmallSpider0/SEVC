"""Selected canonical scientific routines; deployment wrappers are omitted."""
import math

def assess_sampling(samples,window,expected_uuid):
    """Scoped instrument completeness; retain the historical 2.5s quality flag."""
    from sevc.evaluation.workload_performance import integrate_utilization
    if window['sampling_errors'] or not samples:
        raise ValueError('GPU sampling errors or absent samples')
    ticks=[r['monotonic'] for r in samples]
    if (not all(math.isfinite(t) for t in ticks)
            or any(b<=a for a,b in zip(ticks,ticks[1:]))
            or any(r['gpu_uuid']!=expected_uuid for r in samples)
            or any(not math.isfinite(r['utilization_percent']) or not 0<=r['utilization_percent']<=100 for r in samples)):
        raise ValueError('GPU sampling identity, ordering or value invalid')
    measured=integrate_utilization(samples,window['start'],window['end'])
    # Existing GPUSampler waits 1s, then executes two queries with 5s timeout
    # each. This is an instrument budget, not a utilization or science gate.
    instrument_budget=1.+2*5.
    if measured['max_sample_gap_seconds']>instrument_budget:
        raise ValueError('GPU sampling exceeds instrument query budget')
    if not math.isclose(measured['average_percent'],window['utilization']['average_percent'],rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('GPU full-window integral mismatch')
    coarse=[min(window['end'],b)-max(window['start'],a) for a,b in zip(ticks,ticks[1:])
            if b-a>2.5 and min(window['end'],b)>max(window['start'],a)]
    return {'status':'SCOPED_INSTRUMENT_COMPLETENESS_PASS','valid':True,
            'legacy_sampling_valid':window['utilization']['sampling_valid'],
            'instrument_interval_budget_seconds':instrument_budget,
            'max_sample_gap_seconds':measured['max_sample_gap_seconds'],
            'intervals_exceeding_legacy_2_5_seconds':len(coarse),
            'seconds_in_those_intervals':sum(coarse),'samples':len(samples),
            'average_percent':measured['average_percent'],
            'measurement':'left-held utilization samples at actual host receipt timestamps; variable temporal resolution; not continuous kernel busy time',
            'gaps_removed_or_imputed':False}
