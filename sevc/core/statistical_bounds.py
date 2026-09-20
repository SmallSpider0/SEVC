"""Shared exact binomial bounds for protocol calibration and evaluation."""

def clopper_pearson_bound(
    successes: int,
    trials: int,
    *,
    side: str,
    alpha: float,
) -> float:
    """Return an exact one-sided binomial confidence bound.

    Boundary cases deliberately retain non-degenerate uncertainty: zero
    successes has a positive upper bound and all successes has a lower bound
    below one.
    """

    from scipy.stats import beta

    from numbers import Integral
    if (isinstance(successes, bool) or isinstance(trials, bool)
            or not isinstance(successes, Integral) or not isinstance(trials, Integral)):
        raise ValueError("binomial counts must be integers, not rounded observations")
    successes = int(successes)
    trials = int(trials)
    if trials <= 0 or not 0 <= successes <= trials or not 0 < alpha < 1:
        raise ValueError("invalid Clopper-Pearson inputs")
    if side == "upper":
        return 1.0 if successes == trials else float(
            beta.ppf(1.0 - alpha, successes + 1, trials - successes)
        )
    if side == "lower":
        return 0.0 if successes == 0 else float(
            beta.ppf(alpha, successes, trials - successes + 1)
        )
    raise ValueError("Clopper-Pearson side must be lower or upper")

