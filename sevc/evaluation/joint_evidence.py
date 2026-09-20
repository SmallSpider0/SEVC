"""Fail-closed joint-bound inputs. Hash integrity is not a statistical proof."""
import json
import math
from pathlib import Path
from sevc.core.artifacts import sha256_file


def checked_joint_assurance(parameter_evidence, required_events, hard_gates, *,
                            evidence_root=None, certificate=None, required_hard_gates=None):
    reasons=[]
    result={'joint_pass_lower':None,'decision':'HOLD','threshold':.9,
            'formal_launch_ready':False,'F_started':False,'reasons':reasons}
    events=list(required_events)
    gates=list(required_hard_gates or [])
    if not events or len(events)!=len(set(events)):
        reasons.append('empty_or_duplicate_event_registry')
    if set(parameter_evidence)!=set(events):reasons.append('event_evidence_set_mismatch')
    if not gates or len(gates)!=len(set(gates)) or set(hard_gates)!=set(gates):
        reasons.append('missing_or_mismatched_hard_gate_registry')
    if not hard_gates or any(value is not True for value in hard_gates.values()):
        reasons.append('hard_gates_unresolved')
    bounds=[]
    for event in events:
        row=parameter_evidence.get(event,{})
        value=row.get('failure_upper')
        if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=1:
            reasons.append('unknown_or_invalid_bound:'+event);continue
        if row.get('transfer_justified') is not True:reasons.append('transfer_unjustified:'+event)
        bounds.append(value)
    if certificate is None or evidence_root is None:
        reasons.append('missing_common_parameter_certificate')
        return result
    root=Path(evidence_root).resolve()
    def read_bound_file(binding):
        rel=Path(binding['path']);path=(root/rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError('invalid evidence path')
        if sha256_file(path)!=binding['sha256']:raise ValueError('evidence hash drift')
        return json.loads(path.read_text())
    try:
        if certificate.get('schema')!='sevc-joint-parameter-certificate-v1':raise ValueError('certificate schema')
        registry=read_bound_file(certificate['event_registry'])
        if registry['event_ids']!=events or registry['hard_gate_ids']!=gates:raise ValueError('frozen registry mismatch')
        theta=read_bound_file(certificate['theta'])
        if theta['event_ids']!=events:raise ValueError('Theta event coverage mismatch')
        alpha=theta['alpha_allocations']
        if set(alpha)!=set(events) or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in alpha.values()) or sum(alpha.values())>.05:
            raise ValueError('invalid simultaneous error allocation')
        if theta['design_sha256']!=certificate['design']['sha256']:raise ValueError('design identity mismatch')
        read_bound_file(certificate['design'])
        for event in events:
            row=parameter_evidence[event]
            if row['theta_sha256']!=certificate['theta']['sha256']:raise ValueError('events do not share Theta')
            raw=read_bound_file(row['source'])
            if row.get('source_hash')!=row['source']['sha256']:raise ValueError('source binding mismatch')
            if raw.get('event_id')!=event:raise ValueError('raw event mismatch')
        # Scientific derivations are deliberately not accepted from hand-written
        # booleans or arbitrary hashed JSON. Each family needs an executable
        # derivation verifier registered after independent parameter design review.
        # The current 124-event model has no such complete reviewed certificate.
        reasons.append('family_derivation_verifier_not_registered')
    except (KeyError,ValueError,TypeError,OSError,json.JSONDecodeError) as error:
        reasons.append('invalid_common_certificate:'+str(error))
    if not any(x.startswith(('unknown_or_invalid_bound','event_evidence_set','empty_or_duplicate')) for x in reasons):
        result['diagnostic_union_lower']=max(0.,1-math.fsum(bounds))
    return result
