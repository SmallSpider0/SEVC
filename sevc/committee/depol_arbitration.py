"""DePoL JPDC 105056, §4.2/Eqs.1-2, Algorithm 1 and §4.3.

Local subprotocol only. Numeric transfers were not defined in §7.2.
"""
from __future__ import annotations
from sevc.verification.reference_acquisition import commitment


class NativeCommitments:
    """Commit all parties before making any report available to the arbiter."""
    def __init__(self, parties):
        self.parties = tuple(parties)
        self.committed = {}
        self.revealed = {}
        self.events = []
        self.closed = False

    def commit(self, party, digest):
        if self.closed or party not in self.parties or party in self.committed or self.revealed:
            raise ValueError('invalid native commitment phase')
        self.committed[party] = digest
        self.events.append({'phase': 'COMMITTED', 'party': party, 'commitment': digest})

    def reveal(self, party, report):
        if self.closed or set(self.committed) != set(self.parties):
            raise ValueError('all native commitments must precede disclosure')
        if party in self.revealed or commitment([party, report]) != self.committed[party]:
            raise ValueError('address-bound native commitment mismatch')
        self.revealed[party] = report
        self.events.append({'phase': 'REVEALED', 'party': party, 'report': report})

    def expire(self):
        """Terminate incomplete exchange without inventing missing-party penalties."""
        self.closed=True
        missing=[p for p in self.parties if p not in self.revealed]
        row={'phase':'EXPIRED','missing':missing,'status':'HOLD_NATIVE_TIMEOUT_ENDPOINT',
             'trainer_verdict':None,'verifier_reward_eligible':None,
             'numeric_payment':None,'numeric_payment_status':'NOT_DEFINED'}
        self.events.append(row)
        return row


def arbitrate_digests(trainer, reports):
    """Algorithm 1: strict majority of full native digest vectors."""
    m = len(reports)
    eligible = [r is not None and sum(other == r for other in reports if other is not None) > m/2
                for r in reports]
    selected = next((r for r, ok in zip(reports, eligible) if ok), None)
    return {'trainer_verdict': selected == trainer if selected is not None else False,
            'verifier_reward_eligible': eligible, 'trainer_reward_eligible': selected == trainer if selected is not None else False,
            'numeric_payment': None, 'numeric_payment_status': 'NOT_DEFINED'}


def arbitrate_distances(matrices, epsilon):
    """Eq.3: threshold, AND over checkpoints, then majority over vectors."""
    vectors = [[all(row[col] < epsilon for row in matrix) for col in range(len(matrices)+1)]
               for matrix in matrices]
    selected = next((v for v in vectors if vectors.count(v) > len(vectors)/2), None)
    return {'trainer_verdict': None if selected is None else selected[0],
            'trainer_reward_eligible': None if selected is None else selected[0],
            'verifier_reward_eligible': [None]*len(matrices) if selected is None else selected[1:],
            'numeric_payment': None, 'numeric_payment_status': 'NOT_DEFINED',
            'threshold_vectors': vectors, 'majority_vector': selected}
