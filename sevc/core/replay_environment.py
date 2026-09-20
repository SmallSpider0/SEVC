"""Fail-closed environment binding for new server-only replay evidence."""
import hashlib
import json
import platform
from pathlib import Path

POLICY = 'sevc-server-only-v1'
PROFILE = 'registered-server-profile'
GPU_UUID = 'GPU-UNREGISTERED'


class ReplayEnvironmentMismatch(PermissionError):
    """Technical non-adjudication; never an actor's incorrect-verdict evidence."""


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def expected_gpu_uuid(config):
    # Historical evidence retains its original identity. A successor is bound
    # to its own immutable live-captured environment, never a generic GPU.
    if config.get('change_id') != 'experiment-tdsc-server-unlock-evidence-v1':
        return GPU_UUID
    policy = config.get('server_only', {})
    path = Path(policy.get('environment_lock_path', '/nonexistent'))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != policy.get('environment_lock_sha256'):
        raise ReplayEnvironmentMismatch('successor environment lock absent or drifted')
    env = json.loads(path.read_text())
    value = env.get('gpu_uuid', '')
    if env.get('profile') != PROFILE or not value.startswith('GPU-') or len(value) != 40:
        raise ReplayEnvironmentMismatch('successor live GPU identity invalid')
    return value


def require_server_policy(config, profile):
    policy = config.get('server_only', {})
    if (policy.get('policy') != POLICY or policy.get('profile') != PROFILE or
        policy.get('gpu_uuid') != expected_gpu_uuid(config) or profile.get('gpu_uuid') != expected_gpu_uuid(config) or
        profile.get('device') != 'cuda:0' or not policy.get('environment_lock_path') or
        len(policy.get('environment_lock_sha256', '')) != 64):
        raise ReplayEnvironmentMismatch('server-only 225 environment contract required')
    if set(config['datasets']) != {'mnist', 'cifar10', 'cifar100'}:
        raise ReplayEnvironmentMismatch('server-only evidence requires three datasets')


def configure_strict_runtime():
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)


def capture_environment(models, performance, gpu_uuid):
    """Called only after reviewed CUDA preflight and numerical configuration."""
    import torch
    import torchvision
    import subprocess
    import os
    repo = Path(__file__).resolve().parents[2]
    paths = ['sevc/training/engine.py', 'sevc/training/replay_sources.py',
        'sevc/models/factory.py', 'sevc/attacks/strategies.py', 'sevc/core/runtime.py',
        'sevc/core/replay_environment.py', 'sevc/verification/on_demand_service.py',
        'sevc/verification/replay_coupled_probes.py']
    return {'schema':'sevc-replay-environment-v1', 'profile':PROFILE,
        'machine_id':Path('/etc/machine-id').read_text().strip(),
        'system':platform.system(), 'architecture':platform.machine(),
        'python':platform.python_version(), 'torch':str(torch.__version__),
        'torchvision':str(torchvision.__version__), 'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(), 'gpu_uuid':gpu_uuid,
        'gpu_model':torch.cuda.get_device_name(0),
        'gpu_capability':list(torch.cuda.get_device_capability(0)),
        'driver':subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
            '--format=csv,noheader'], text=True).strip(),
        'torch_build_sha256':hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        'device':'cuda:0', 'cpu_threads':torch.get_num_threads(),
        'interop_threads':torch.get_num_interop_threads(),
        'deterministic':torch.are_deterministic_algorithms_enabled(),
        'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'dtype':str(torch.get_default_dtype()), 'autocast':torch.is_autocast_enabled(),
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'models':models, 'performance':performance,
        'semantic_code':{p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}}


def validate_environment(config, actual):
    policy = config['server_only']
    path = Path(policy['environment_lock_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != policy['environment_lock_sha256']:
        raise ReplayEnvironmentMismatch('server environment lock hash mismatch')
    expected = json.loads(path.read_text())
    if (actual != expected or actual['system'] != 'Linux' or actual['device'] != 'cuda:0' or
        actual['gpu_uuid'] != expected_gpu_uuid(config) or not actual['deterministic'] or actual['warn_only']):
        raise ReplayEnvironmentMismatch('server replay environment differs from frozen identity')
    return identity(actual)


def bind_source(proof_hash, environment_id):
    return {'replay_environment_id':environment_id,
            'environment_proof_binding':identity([POLICY, proof_hash, environment_id])}


def require_source(row, environment_id, bridge=None):
    if environment_id is None:
        return  # Historical non-submission interfaces retain their old semantics.
    origin = row.get('replay_environment_id')
    accepted = environment_id
    if bridge and bridge.get('successor_id') == environment_id and origin == bridge.get('predecessor_id'):
        accepted = origin
    expected = bind_source(row['proof_sha256'], accepted)
    if any(row.get(k) != v for k, v in expected.items()):
        raise ReplayEnvironmentMismatch('source has missing or incompatible server environment binding')


def require_sources(sources, environment_id, bridge=None):
    if environment_id is None:
        return
    for source in sources:
        require_source({**source.recipe, **source.hashes}, environment_id, bridge)


def require_pending_calibration(pending, environment_id, bridge=None):
    if environment_id is None:
        return
    if pending.get('source_device') != 'cuda:0' or pending.get('replay_environment_id') not in accepted_environments(environment_id, bridge):
        raise ReplayEnvironmentMismatch('calibration cannot import historical or foreign environment evidence')
    require_sources(pending['sources'].values(), environment_id, bridge)


def require_continuation(manifest, completed_rows, environment_id, bridge=None):
    if environment_id is None:
        return
    for group in manifest['source_groups'].values():
        if group.get('source_device') != 'cuda:0' or group['row'].get('replay_environment_id') not in accepted_environments(environment_id, bridge):
            raise ReplayEnvironmentMismatch('continuation contains historical or foreign source group')
        for row in group['row']['sources']:
            require_source(row, environment_id, bridge)
    for row in completed_rows:
        if (row.get('replay_environment_id') not in accepted_environments(environment_id, bridge) or
            row.get('runtime_device') != 'cuda:0' or row.get('source_origin_device') != 'cuda:0'):
            raise ReplayEnvironmentMismatch('continuation contains foreign completed cost or service evidence')


def accepted_environments(environment_id, bridge=None):
    result = {environment_id}
    if bridge and bridge.get('successor_id') == environment_id:
        result.add(bridge['predecessor_id'])
    return result


def validate_implementation_bridge(config, actual, *, artifact_root=None):
    """Explicit, hash-locked compatibility; never relabel a historical source."""
    policy = config.get('server_only', {})
    if not policy.get('bridge_lock_path'):
        return None
    if config.get('change_id') != 'experiment-tdsc-server-unlock-evidence-v1':
        raise ReplayEnvironmentMismatch('bridge is restricted to reviewed successor')
    path = Path(policy['bridge_lock_path'])
    if artifact_root is not None:
        path = Path(artifact_root)/path.name
    if hashlib.sha256(path.read_bytes()).hexdigest() != policy.get('bridge_lock_sha256'):
        raise ReplayEnvironmentMismatch('bridge lock drift')
    lock = json.loads(path.read_text())
    before = lock['predecessor_environment']
    if (not lock.get('ready') or lock.get('revision') != 4 or
            lock['successor_id'] != identity(actual) or lock['predecessor_id'] != identity(before)):
        raise ReplayEnvironmentMismatch('bridge identity or review mismatch')
    for key in set(before) | set(actual):
        if key not in {'semantic_code','performance'} and before.get(key) != actual.get(key):
            raise ReplayEnvironmentMismatch('bridge numerical/platform drift: '+key)
    allowed_code = {'sevc/verification/on_demand_service.py',
        'sevc/verification/replay_coupled_probes.py', 'sevc/core/replay_environment.py'}
    changed = {k for k in set(before['semantic_code']) | set(actual['semantic_code'])
               if before['semantic_code'].get(k) != actual['semantic_code'].get(k)}
    if not changed <= allowed_code or sorted(changed) != lock['changed_code_paths']:
        raise ReplayEnvironmentMismatch('bridge changes numerical implementation')
    changed_perf = {k for k in set(before['performance']) | set(actual['performance'])
                    if before['performance'].get(k) != actual['performance'].get(k)}
    if changed_perf != {'compact_source_headers','resident_delivery_bytes'} or (
        actual['performance']['compact_source_headers'] is not True or
        actual['performance']['resident_delivery_bytes'] != 3*1024**3):
        raise ReplayEnvironmentMismatch('bridge performance scope drift')
    evidence_path = Path(lock['engineering_evidence_path'])
    if artifact_root is not None:
        evidence_path = Path(artifact_root)/evidence_path.name
    if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != lock['engineering_evidence_sha256']:
        raise ReplayEnvironmentMismatch('bridge equivalence evidence drift')
    evidence = json.loads(evidence_path.read_text())
    if evidence.get('status') != 'PASS_ENGINEERING_EQUIVALENCE_AND_ONLINE_COST_GATE' or {
        x['dataset'] for x in evidence['datasets'] if x['public_tasks_byte_identical']} != {'mnist','cifar10','cifar100'}:
        raise ReplayEnvironmentMismatch('three-dataset equivalence absent')
    return {'predecessor_id':lock['predecessor_id'],'successor_id':lock['successor_id'],
            'bridge_lock_sha256':policy['bridge_lock_sha256']}
