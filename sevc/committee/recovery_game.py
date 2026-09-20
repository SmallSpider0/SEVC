"""Worst-case recoverable-value recovery on capacity-one rosters.

A game instance holds, per identity, a bitmask of compatible jobs and, per job, the number of
service-supported reports it needs.  Any non-PASS outcome fails (and quarantines) an identity; the
failure allowance ``k`` counts failed identities.  Only public state enters every decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import itertools
import random

from sevc.committee.formation import majority_success_probability

EXACT = 'value-preserving-exact'
RC = 'value-preserving-rc'
MATCHING_WAVE = 'matching-wave-ecs'
ADVERSARIES = ('static', 'adaptive-scarcity', 'adaptive-first')


def required_passes(p_min: float, rho: float, limit: int = 99) -> int | None:
    """Smallest odd decision-set size whose majority reaches rho at correctness p_min."""
    for size in range(1, limit+1, 2):
        if majority_success_probability([p_min]*size) >= rho:
            return size
    return None


@dataclass(frozen=True)
class Game:
    compat: tuple[int, ...]
    need: tuple[int, ...]
    allowance: int

    def __post_init__(self):
        jobs = len(self.need)
        if not jobs or any(q < 1 for q in self.need) or self.allowance < 0:
            raise ValueError('invalid recovery game')
        if any(c < 0 or c >> jobs for c in self.compat):
            raise ValueError('compatibility outside registered jobs')


@dataclass
class PublicState:
    unused: int
    passes: list[int]
    failures: int = 0
    attempts: list[int] = field(default_factory=list)
    rounds: int = 0

    @classmethod
    def initial(cls, game: Game) -> 'PublicState':
        jobs = len(game.need)
        return cls((1 << len(game.compat))-1, [0]*jobs, 0, [0]*jobs, 0)

    def unresolved(self, game: Game) -> list[int]:
        return [j for j, q in enumerate(game.need) if self.passes[j] < q]

    def completed(self, game: Game) -> int:
        return sum(p >= q for p, q in zip(self.passes, game.need))


def _bits(mask: int):
    i = 0
    while mask:
        if mask & 1:
            yield i
        mask >>= 1
        i += 1


# ---------------------------------------------------------------- exact value (reference)

@lru_cache(maxsize=64)
def exact_solver(compat: tuple[int, ...], need: tuple[int, ...]):
    """Min-max completed-job value; ties: worst, optimistic, flexibility, job, identity."""
    n, jobs = len(compat), len(need)
    if n > 14 or jobs > 3:
        raise ValueError('exact solver is registered for at most 14 identities and 3 jobs')

    @lru_cache(maxsize=None)
    def solve(remaining: int, counts: tuple[int, ...], failures: int):
        completed = sum(c >= q for c, q in zip(counts, need))
        if completed == jobs or not remaining:
            return completed, None
        best = None
        for i in range(n):
            if not remaining >> i & 1:
                continue
            rest = remaining ^ (1 << i)
            for j in range(jobs):
                if counts[j] >= need[j] or not compat[i] >> j & 1:
                    continue
                updated = counts[:j]+(counts[j]+1,)+counts[j+1:]
                success = solve(rest, updated, failures)[0]
                worst = min(success, solve(rest, counts, failures-1)[0]) if failures else success
                optimistic = solve(rest, updated, 0)[0] if failures else success
                key = (-worst, -optimistic, compat[i].bit_count(), j, i)
                if best is None or key < best[0]:
                    best = key, (j, i)
        return (completed, None) if best is None else (-best[0][0], best[1])
    return solve


def exact_round(game: Game, state: PublicState, memory: dict):
    solve = exact_solver(game.compat, game.need)
    b = max(0, game.allowance-state.failures)
    counts = tuple(min(p, q) for p, q in zip(state.passes, game.need))
    value, action = solve(state.unused, counts, b)
    memory.setdefault('root_value', value)
    step = {'guaranteed_count': value, 'remaining_failure_allowance': b,
            'action': list(action) if action else None}
    return ([] if action is None else [action]), step


# ---------------------------------------------------------------- reserve-cover certificate

def _coverage(game: Game, mask: int, jobs) -> dict[int, int]:
    cover = {j: 0 for j in jobs}
    for i in _bits(mask):
        for j in jobs:
            if game.compat[i] >> j & 1:
                cover[j] += 1
    return cover


def _construct(game: Game, mask: int, deficits: dict[int, int], b: int, reserve_jobs=None,
               waiting=()):
    """Disjoint primaries for every job in ``deficits``; each job in ``reserve_jobs`` (default: all)
    keeps at least b compatible unassigned identities.

    Greedy: serve the tightest job first; among identities that keep every slack nonnegative, take
    the one least useful to ``waiting`` jobs, then the one hurting the tightest other job least, so
    released reserves stay useful to jobs served later.  Returns {job: [identities]} or None.
    """
    jobs = sorted(deficits)
    reserve = set(jobs if reserve_jobs is None else reserve_jobs)
    cover = _coverage(game, mask, jobs)
    pending = dict(deficits)
    slack = {h: cover[h]-pending[h]-(b if h in reserve else 0) for h in jobs}
    if any(s < 0 for s in slack.values()):
        return None
    avail = mask
    plan = {j: [] for j in jobs}
    while any(pending.values()):
        j = min((h for h in jobs if pending[h]), key=lambda h: (slack[h], h))
        best = None
        for i in _bits(avail):
            if not game.compat[i] >> j & 1:
                continue
            affected = [slack[h] for h in jobs if h != j and game.compat[i] >> h & 1]
            low = min(affected) if affected else 1 << 30
            if low <= 0:
                continue
            useful = sum(game.compat[i] >> h & 1 for h in waiting)
            key = (useful, -low, len(affected), game.compat[i].bit_count(), i)
            if best is None or key < best[0]:
                best = key, i
        if best is None:
            return None
        i = best[1]
        avail ^= 1 << i
        plan[j].append(i)
        pending[j] -= 1
        for h in jobs:
            if h != j and game.compat[i] >> h & 1:
                slack[h] -= 1
    return plan


def certify(game: Game, state: PublicState, base: tuple[int, ...] = ()):
    """Grow a guaranteed job set from ``base``; soundness does not need an optimal search."""
    b = max(0, game.allowance-state.failures)
    unresolved = state.unresolved(game)
    deficit = {j: game.need[j]-state.passes[j] for j in unresolved}
    guaranteed = [j for j in base if j in deficit]
    plan = _construct(game, state.unused, {j: deficit[j] for j in guaranteed}, b, guaranteed,
                      [j for j in unresolved if j not in guaranteed])
    if plan is None:
        raise AssertionError('previously guaranteed jobs lost their reserve witness')
    cover = _coverage(game, state.unused, unresolved)
    for j in sorted((j for j in unresolved if j not in guaranteed),
                    key=lambda j: (deficit[j], -cover[j], j)):
        trial = _construct(game, state.unused, {h: deficit[h] for h in guaranteed+[j]}, b,
                           guaranteed+[j], [h for h in unresolved if h not in guaranteed and h != j])
        if trial is not None:
            guaranteed.append(j)
            plan = trial
    guaranteed.sort()
    return {'guaranteed_jobs': guaranteed, 'primaries': {j: plan[j] for j in guaranteed},
            'guaranteed_count': state.completed(game)+len(guaranteed),
            'remaining_failure_allowance': b}


def reserve_invariant(game: Game, mask: int, guaranteed, b: int) -> bool:
    cover = _coverage(game, mask, guaranteed)
    return all(cover[j] >= b for j in guaranteed)


def rc_round(game: Game, state: PublicState, memory: dict):
    """All guaranteed deficits in parallel; other jobs only if the reserve invariant survives."""
    in_scope = state.failures <= game.allowance
    base = tuple(memory.get('guaranteed_jobs', ())) if in_scope else ()
    cert = certify(game, state, base)
    previous = memory.get('guaranteed_count')
    if in_scope and previous is not None and cert['guaranteed_count'] < previous:
        raise AssertionError('value-preserving policy lowered its guaranteed count')
    memory.setdefault('root_certificate', cert)
    memory['guaranteed_jobs'] = cert['guaranteed_jobs']
    memory['guaranteed_count'] = cert['guaranteed_count']
    b = cert['remaining_failure_allowance']
    guaranteed = cert['guaranteed_jobs']
    deficit = {j: game.need[j]-state.passes[j] for j in state.unresolved(game)}
    plan, extra = cert['primaries'], []
    cover = _coverage(game, state.unused, deficit)
    # Second objective (optimistic value): serve further jobs now when G keeps its reserve.
    for j in sorted((j for j in deficit if j not in guaranteed),
                    key=lambda j: (deficit[j], -cover[j], j)):
        trial = _construct(game, state.unused, {h: deficit[h] for h in guaranteed+extra+[j]}, b,
                           guaranteed, [h for h in deficit if h not in guaranteed+extra+[j]])
        if trial is not None:
            extra.append(j)
            plan = trial
    actions = [(j, i) for j in guaranteed for i in plan[j]]
    others = [(j, i) for j in sorted(extra) for i in plan[j]]
    reserve = state.unused
    for _, i in actions+others:
        reserve ^= 1 << i
    if in_scope and not reserve_invariant(game, reserve, guaranteed, b):
        raise AssertionError('round plan broke the reserve invariant')
    step = {'guaranteed_jobs': guaranteed, 'guaranteed_count': cert['guaranteed_count'],
            'remaining_failure_allowance': b, 'within_allowance': in_scope,
            'guaranteed_actions': [list(a) for a in actions],
            'other_actions': [list(a) for a in others]}
    return actions+others, step


# ---------------------------------------------------------------- current-ECS matching baseline

def _augment(game, j, avail, owner, seen):
    for i in _bits(avail):
        if not game.compat[i] >> j & 1 or i in seen:
            continue
        seen.add(i)
        if i not in owner or _augment(game, owner[i], avail, owner, seen):
            owner[i] = j
            return True
    return False


def slot_matching(game: Game, avail: int, slots: list[int]) -> dict[int, int]:
    """Deterministic maximum matching of job slots (in order) to available identities."""
    owner: dict[int, int] = {}
    for j in slots:
        _augment(game, j, avail, owner, set())
    return owner


def matching_round(game: Game, state: PublicState, memory: dict):
    unresolved = state.unresolved(game)
    if not memory.get('primaries_done'):
        memory['primaries_done'] = True
        slots = [j for j in unresolved for _ in range(game.need[j]-state.passes[j])]
    else:
        slots = sorted(unresolved, key=lambda j: (game.need[j]-state.passes[j], j))
    owner = slot_matching(game, state.unused, slots)
    actions = sorted(((j, i) for i, j in owner.items()), key=lambda a: (a[0], a[1]))
    return actions, {'actions': [list(a) for a in actions]}


POLICIES = {EXACT: exact_round, RC: rc_round, MATCHING_WAVE: matching_round}


# ---------------------------------------------------------------- adversaries and simulator

class Adversary:
    def __init__(self, kind: str, budget: int, failed: frozenset[int] = frozenset()):
        if kind not in ADVERSARIES:
            raise ValueError('unregistered adversary')
        self.kind, self.budget, self.static = kind, budget, frozenset(failed)
        self.failed: set[int] = set()

    def outcomes(self, game: Game, state: PublicState, actions):
        if self.kind == 'static':
            result = [i not in self.static for _, i in actions]
        else:
            result = [True]*len(actions)
            target = None
            if self.kind == 'adaptive-scarcity':
                jobs = sorted({j for j, _ in actions})
                cover = _coverage(game, state.unused, jobs)
                target = min(jobs, key=lambda j: (cover[j]-(game.need[j]-state.passes[j]), j),
                             default=None)
            for n, (j, i) in enumerate(actions):
                if len(self.failed) >= self.budget:
                    break
                if self.kind == 'adaptive-first' or j == target:
                    result[n] = False
                    self.failed.add(i)
        for ok, (_, i) in zip(result, actions):
            if not ok:
                self.failed.add(i)
        return result


def simulate(game: Game, policy: str, adversary: Adversary, *, max_rounds: int = 64,
             attempt_cap: int | None = None):
    select = POLICIES[policy]
    state, memory, rounds = PublicState.initial(game), {}, []
    while state.rounds < max_rounds and state.unresolved(game):
        actions, step = select(game, state, memory)
        actions = [(j, i) for j, i in actions
                   if attempt_cap is None or state.attempts[j] < attempt_cap]
        if not actions:
            break
        outcomes = adversary.outcomes(game, state, actions)
        for (j, i), ok in zip(actions, outcomes):
            state.unused &= ~(1 << i)
            state.attempts[j] += 1
            if ok:
                state.passes[j] = min(game.need[j], state.passes[j]+1)
            else:
                state.failures += 1
        state.rounds += 1
        rounds.append({'step': step, 'actions': [list(a) for a in actions],
                       'outcomes': outcomes})
    root = (memory.get('root_certificate', {}).get('guaranteed_count')
            if policy == RC else memory.get('root_value'))
    return {'policy': policy, 'completed_jobs': state.completed(game), 'rounds': state.rounds,
            'failed_identities': sorted(adversary.failed), 'root_guaranteed_count': root,
            'attempts': sum(state.attempts), 'trace': rounds}


# ---------------------------------------------------------------- offline maximum

def feasible_jobs(game: Game, avail: int, jobs) -> bool:
    slots = [j for j in jobs for _ in range(game.need[j])]
    return len(slot_matching(game, avail, slots)) == len(slots)


def offline_maximum(game: Game, failed, cap_checks: int = 20000):
    """Most jobs fully servable from non-failed identities; None if the check cap is exceeded."""
    avail = ((1 << len(game.compat))-1) & ~sum(1 << i for i in failed)
    cover = _coverage(game, avail, range(len(game.need)))
    viable = [j for j in range(len(game.need)) if cover[j] >= game.need[j]]
    checks = 0
    for size in range(len(viable), 0, -1):
        for subset in itertools.combinations(viable, size):
            checks += 1
            if checks > cap_checks:
                return None
            if feasible_jobs(game, avail, subset):
                return size
    return 0


def uniform_failures(n: int, k: int, rng: random.Random) -> frozenset[int]:
    return frozenset(rng.sample(range(n), min(k, n)))
