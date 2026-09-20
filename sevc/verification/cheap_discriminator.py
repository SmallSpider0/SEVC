"""Public-proof-only features and frozen history selection; no owner truth input."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import numpy as np

FEATURE_SCHEMA='public-state-strided512-four-step-v1'
METRICS=('state_rms','update_rms','momentum_rms','update_max','update_mean',
         'update_per_batch','relative_update','sgd_residual','update_cosine')
FEATURE_NAMES=('step_count','batch_size','learning_rate','momentum')+tuple(
    f'step{i}/{m}' for i in range(4) for m in METRICS)


def sample_coordinates(tensor, count):
    """Sample logical flattened coordinates without copying a strided full tensor."""
    import torch
    t=tensor.detach();stride=max(1,(t.numel()+count-1)//count)
    if t.is_contiguous():return t.view(-1)[::stride].clone()
    flat=torch.arange(0,t.numel(),stride,device=t.device)
    coordinates=[]
    for size in reversed(t.shape):
        coordinates.append(flat % size);flat=flat//size
    return t[tuple(reversed(coordinates))]


def public_features(proof, *, coordinates=512):
    """Fixed strided coordinates of delivered states; no forward/backward or path reads."""
    import torch
    if coordinates not in (32,512):raise ValueError("unregistered coordinate budget")
    if len(proof.checkpoints)!=4 or len(proof.optimizer_checkpoints)!=4:
        raise ValueError('tiny feature schema requires exactly four steps')
    size=int(proof.batches[0][0].shape[0]); values=[4,size,proof.learning_rate,proof.momentum]
    previous=proof.initial_state; last_delta=None
    with torch.no_grad():
        for state,momentum in zip(proof.checkpoints,proof.optimizer_checkpoints):
            ws=[];ds=[];ms=[]
            for k in sorted(momentum):
                w0=sample_coordinates(previous[k],coordinates);w1=sample_coordinates(state[k],coordinates);m=sample_coordinates(momentum[k],coordinates)
                ws.append(w0.float().cpu());ds.append((w1-w0).float().cpu());ms.append(m.float().cpu())
            w,d,m=torch.cat(ws),torch.cat(ds),torch.cat(ms)
            rms=lambda a:float(a.square().mean().sqrt())
            norm=rms(d);cos=0. if last_delta is None else float(torch.nn.functional.cosine_similarity(d,last_delta,dim=0))
            values.extend([rms(w),norm,rms(m),float(d.abs().max()),float(d.mean()),norm/size,
                           norm/(rms(w)+1e-12),float((d+proof.learning_rate*m).abs().max()),cos])
            previous=state;last_delta=d
    if len(values)!=len(FEATURE_NAMES) or not np.isfinite(values).all():raise ValueError('invalid public feature vector')
    return values


def select_from_history(history, features, task_ids, *, quota, models, random_state):
    """Test labels are absent by construction; fixed top-k under disclosed development labels."""
    from sevc.verification.adaptive_recognizer import fit_predict_probe_probability_v2
    if history.get('disclosed') is not True or history.get('split')!='development':
        raise PermissionError('only disclosed development history may train the recognizer')
    if set(history['task_ids']) & set(task_ids):raise ValueError('history/test task overlap')
    if not 0 <= quota <= len(task_ids):raise ValueError('invalid replay quota')
    n=len(history['features']); x=np.asarray(history['features']+features,dtype=float)
    y=np.asarray(history['labels']+[0]*len(features),dtype=int) # dummy labels never selected for fitting
    probabilities,receipt=fit_predict_probe_probability_v2(matrix=x,labels=y,
        history_indices=list(range(n)),test_indices=list(range(n,len(x))),observation_budget=n,
        random_state=random_state,component_models=models)
    order=sorted(range(len(task_ids)),key=lambda i:(-float(probabilities[i]),task_ids[i]))
    return [task_ids[i] for i in order[:quota]],probabilities.tolist(),asdict(receipt)


class CheapSelector:
    """An actor capability containing only disclosed development data and public features."""
    def __init__(self,history,models,quota=32):
        self.history=history; self.models=models; self.quota=quota; self.receipt=None

    def __call__(self,tasks,clock):
        from sevc.verification.paid_replay_service import StoredReplayProof
        features=[]
        for t in tasks:
            proof=t.proof
            if isinstance(proof,StoredReplayProof):proof,_=clock.call('recognizer_payload_read','verifier',proof.load)
            f,_=clock.call('recognizer_public_features','verifier',public_features,proof)
            features.append(f)
        ids=[t.task_id for t in tasks]
        (selected,scores,fit),cost=clock.call('recognizer_fit_and_predict','verifier',select_from_history,
            self.history,features,ids,quota=self.quota,models=self.models,random_state=20260916)
        self.receipt={'schema':FEATURE_SCHEMA,'feature_names':FEATURE_NAMES,'features':features,'task_ids':ids,
            'scores':scores,'selected':selected,'quota':self.quota,'fit_receipt':fit,
            'fit_and_predict_seconds':cost,'training_charge':'refit charged on every assignment; no free amortization',
            'history_sha256':hashlib.sha256(json.dumps(self.history,sort_keys=True).encode()).hexdigest()}
        return frozenset(selected)


def history_digest(history):
    return hashlib.sha256(json.dumps(history,sort_keys=True).encode()).hexdigest()


def select_development_model(development, validation, models, random_state=20260916):
    """Eight fixed candidates; development fits, validation selects, no test input."""
    from sklearn.metrics import roc_auc_score
    from sevc.verification.adaptive_recognizer import build_probe_classifiers
    import time
    for h in (development,validation):
        if not h.get('disclosed'):raise PermissionError('undisclosed selection history')
    if development['split']!='development' or validation['split']!='validation':
        raise PermissionError('selection requires development and validation only')
    if set(development['task_ids']) & set(validation['task_ids']):raise ValueError('selection split overlap')
    receipts=[];fitted={}
    names=('logistic','hist_gradient_boosting','extra_trees','ensemble')
    y=np.asarray(development['labels']);v=np.asarray(validation['labels'])
    if set(y)!={0,1} or set(v)!={0,1}:raise ValueError('development labels need both classes')
    for variant in ('stride512','stride32'):
        x=np.asarray(development['feature_variants'][variant]);z=np.asarray(validation['feature_variants'][variant])
        components=build_probe_classifiers(models,random_state);scores=[];fit_seconds=[];predict_seconds=[]
        for model in components:
            began=time.perf_counter();model.fit(x,y);fit_seconds.append(time.perf_counter()-began)
            began=time.perf_counter();scores.append(model.predict_proba(z)[:,1]);predict_seconds.append(time.perf_counter()-began)
        scores.append(np.mean(scores,axis=0))
        for rank,(name,p) in enumerate(zip(names,scores)):
            picked=sorted(range(len(v)),key=lambda i:(-float(p[i]),validation['task_ids'][i]))[:32]
            receipt={'feature_variant':variant,'model':name,'reject_coverage':int(v[picked].sum()),
                     'reject_count':int(v.sum()),'auc':float(roc_auc_score(v,p)),
                     'fit_seconds':sum(fit_seconds) if rank==3 else fit_seconds[rank],
                     'validation_predict_seconds':sum(predict_seconds) if rank==3 else predict_seconds[rank]}
            receipts.append(receipt);fitted[variant,name]=components if rank==3 else (components[rank],)
    chosen=min(receipts,key=lambda r:(-r['reject_coverage'],-r['auc'],r['feature_variant']!='stride32',names.index(r['model'])))
    report={'candidates':receipts,'selected':dict(chosen),'development_sha256':history_digest(development),
            'validation_sha256':history_digest(validation),'test_used':False,
            'training_task_ids':development['task_ids'],'selection_task_ids':validation['task_ids']}
    return fitted[chosen['feature_variant'],chosen['model']],report


class PersistentCheapSelector:
    """Frozen model with a bounded, authenticated, assignment-local public cache."""
    def __init__(self,models,selection,cache_limit_bytes):
        from collections import OrderedDict
        self.models=models;self.selection=selection;self.limit=int(cache_limit_bytes)
        self.cache=OrderedDict();self.used=0;self.peak=0;self.hits=0;self.misses=0;self.receipt=None

    def _remember(self,task,proof):
        from sevc.core.scratch_budget import tensor_bytes
        size=tensor_bytes(proof)
        if size>self.limit:return
        while self.used+size>self.limit and self.cache:
            _,(_,_,old_size)=self.cache.popitem(last=False);self.used-=old_size
        self.cache[task.task_id]=(task.proof,proof,size);self.used+=size;self.peak=max(self.peak,self.used)

    def _predict(self,features):
        p=np.mean([m.predict_proba(np.asarray(features))[:,1] for m in self.models],axis=0)
        if not np.isfinite(p).all():raise ValueError('nonfinite attack prediction')
        return p.tolist()

    def __call__(self,tasks,clock):
        from sevc.verification.paid_replay_service import StoredReplayProof
        ids=[t.task_id for t in tasks]
        if set(ids)&set(self.selection['training_task_ids']+self.selection['selection_task_ids']):
            raise ValueError('test/history task overlap')
        coordinates=int(self.selection['selected']['feature_variant'].removeprefix('stride'));features=[]
        for task in tasks:
            proof=task.proof
            if isinstance(proof,StoredReplayProof):
                proof,_=clock.call('recognizer_payload_read','verifier',proof.load)
                clock.call('recognizer_cache_admission','verifier',self._remember,task,proof)
            f,_=clock.call('recognizer_public_features','verifier',public_features,proof,coordinates=coordinates)
            features.append(f)
        scores,predict_seconds=clock.call('recognizer_predict','verifier',self._predict,features)
        selected=sorted(ids,key=lambda t:(-scores[ids.index(t)],t))[:32]
        for task_id in list(self.cache):
            if task_id not in selected:
                _,_,size=self.cache.pop(task_id);self.used-=size
        self.receipt={'schema':'public-state-strided-four-step-v2','feature_names':FEATURE_NAMES,
            'features':features,'task_ids':ids,'scores':scores,'selected':selected,'quota':32,
            'feature_variant':self.selection['selected']['feature_variant'],'model':self.selection['selected']['model'],
            'fit_and_predict_seconds':predict_seconds,'online_fit_count':0,'training_charge':'startup recorded separately',
            'history_sha256':self.selection['development_sha256'],
            'selection_sha256':hashlib.sha256(json.dumps(self.selection,sort_keys=True).encode()).hexdigest()}
        return frozenset(selected)

    def load_for_replay(self,task):
        found=self.cache.pop(task.task_id,None)
        if found is None:
            self.misses+=1;return task.proof.load()
        handle,proof,size=found;self.used-=size
        if handle is not task.proof:raise ValueError('public cache handle mismatch')
        # Cached mappings are valid only while authenticated file identity is unchanged.
        if handle.access_profile not in {'authenticated-mmap','owned-mmap'}:
            from sevc.core.artifacts import sha256_file
            if sha256_file(handle.path)!=handle.sha256:raise ValueError('cached payload identity drift')
        else:handle.mapped_file.check_identity()
        self.hits+=1;return proof

    def finish(self):
        if self.receipt is not None:
            self.receipt['payload_cache']={'hits':self.hits,'misses':self.misses,'limit_bytes':self.limit,
                'peak_logical_tensor_bytes':self.peak,'scope':'one assignment; public payloads only'}
        self.cache.clear();self.used=0


def collect_public_history(unit, job, clock, *, improved):
    """Collect public features; keep labels private until the caller closes its epoch."""
    from sevc.verification.paid_replay_service import StoredReplayProof
    features = []; variants = {'stride512': [], 'stride32': []}
    for task in job.tasks:
        if isinstance(task.proof, StoredReplayProof):
            proof, _ = clock.call('development_payload_read', 'offline', task.proof.load)
        else:
            proof = task.proof
        value, _ = clock.call('development_feature_extraction', 'offline', public_features, proof)
        features.append(value)
        if improved:
            variants['stride512'].append(value)
            compact, _ = clock.call('development_feature_extraction_stride32', 'offline', public_features, proof, coordinates=32)
            variants['stride32'].append(compact)
    answers = dict(job.probes.answers)
    history = {'split': unit['split'], 'seed': unit['seed'], 'block': unit['block'],
               'task_ids': [t.task_id for t in job.tasks], 'features': features,
               'labels': [int(answers.get(t.task_id, True) is False) for t in job.tasks],
               'target': 'reject-challenge', 'disclosed': False}
    if improved:
        history['feature_variants'] = variants
    return history


def publish_history(history, path):
    """Call only after epoch closure; no test history may be published for fitting."""
    from sevc.core.artifacts import write_json
    if history['split'] not in ('development', 'validation'):
        raise PermissionError('test labels cannot enter training history')
    history['disclosed'] = True
    write_json(path, history)


def select_development_models(development, validation, models, random_state=20260916):
    """Fit pooled development; rank by SUM of per-service validation top32 coverage."""
    from sklearn.metrics import roc_auc_score
    from sevc.verification.adaptive_recognizer import build_probe_classifiers
    import time
    if not development or not validation:
        raise ValueError('empty selection history')
    seen = set()
    for split, histories in (('development', development), ('validation', validation)):
        for h in histories:
            if h.get('split') != split or h.get('disclosed') is not True:
                raise PermissionError('only disclosed development/validation histories allowed')
            ids = h['task_ids']
            if len(ids) != 40 or len(set(ids)) != 40 or seen.intersection(ids):
                raise ValueError('duplicate, overlapping or incomplete history tasks')
            if len(h['labels']) != 40 or set(h['labels']) != {0, 1}:
                raise ValueError('invalid history labels')
            if any(len(h['feature_variants'][v]) != 40 for v in ('stride512', 'stride32')):
                raise ValueError('incomplete history features')
            seen.update(ids)
    names = ('logistic', 'hist_gradient_boosting', 'extra_trees', 'ensemble')
    train_ids = [i for h in development for i in h['task_ids']]
    val_ids = [i for h in validation for i in h['task_ids']]
    y = np.asarray([v for h in development for v in h['labels']])
    v = np.asarray([v for h in validation for v in h['labels']])
    receipts = []; fitted = {}
    for variant in ('stride512', 'stride32'):
        x = np.asarray([f for h in development for f in h['feature_variants'][variant]])
        z = np.asarray([f for h in validation for f in h['feature_variants'][variant]])
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError('nonfinite history features')
        components = build_probe_classifiers(models, random_state)
        scores = []; fit_seconds = []; predict_seconds = []
        for model in components:
            start = time.perf_counter(); model.fit(x, y); fit_seconds.append(time.perf_counter()-start)
            start = time.perf_counter(); scores.append(model.predict_proba(z)[:, 1]); predict_seconds.append(time.perf_counter()-start)
        scores.append(np.mean(scores, axis=0))
        for rank, (name, probabilities) in enumerate(zip(names, scores)):
            coverage = []
            for offset in range(0, len(v), 40):
                picked = sorted(range(offset, offset+40), key=lambda i: (-float(probabilities[i]), val_ids[i]))[:32]
                coverage.append(int(v[picked].sum()))
            receipts.append({'feature_variant': variant, 'model': name, 'reject_coverage': sum(coverage),
                'per_service_reject_coverage': coverage, 'reject_count': int(v.sum()),
                'auc': float(roc_auc_score(v, probabilities)),
                'fit_seconds': sum(fit_seconds) if rank == 3 else fit_seconds[rank],
                'validation_predict_seconds': sum(predict_seconds) if rank == 3 else predict_seconds[rank]})
            fitted[variant, name] = components if rank == 3 else (components[rank],)
    chosen = min(receipts, key=lambda r: (-r['reject_coverage'], -r['auc'], r['feature_variant'] != 'stride32', names.index(r['model'])))
    report = {'candidates': receipts, 'selected': dict(chosen),
        'development_sha256': history_digest(development), 'validation_sha256': history_digest(validation),
        'test_used': False, 'training_task_ids': train_ids, 'selection_task_ids': val_ids,
        'development_services': len(development), 'validation_services': len(validation),
        'selection_rule': 'sum per-service top32 coverage, pooled AUC, stride32, model order'}
    return fitted[chosen['feature_variant'], chosen['model']], report
