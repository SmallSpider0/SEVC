"""Hash-first cleanup of owned derivatives, tolerating open NFS tombstones."""
import errno
import gc
import os
import shutil
from pathlib import Path
from sevc.core.artifacts import sha256_file


def remove_owned_scratch(path, emit):
    path=Path(path)
    if path.is_symlink():
        raise ValueError("scratch cleanup cannot follow symlinks")
    if not path.exists():
        return True
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("scratch cleanup cannot follow symlinks")
    files=[{'path':str(p),'sha256':sha256_file(p),'bytes':p.stat().st_size}
           for p in sorted(path.rglob('*')) if p.is_file()]
    emit({'event':'SCRATCH_HASH_SEAL_BEFORE_RELEASE','path':str(path),'files':files})
    try:
        shutil.rmtree(path)
    except OSError as error:
        remaining=[p for p in path.rglob('*') if p.is_file()]
        if error.errno not in {errno.ENOTEMPTY,errno.EBUSY} or not remaining or any(not p.name.startswith('.nfs') for p in remaining):
            raise
        emit({'event':'SCRATCH_NFS_OPEN_FILE_DEFERRED','path':str(path),
              'remaining_files':[str(p) for p in remaining],'scientific_unit_repeated':False})
        return False
    return True


def release_scratch(study, path):
    pending=getattr(study,'pending_scratch_cleanup',set())
    pending.add(str(path));gc.collect()
    study.pending_scratch_cleanup={p for p in sorted(pending)
        if not remove_owned_scratch(p,study.events)}


def registered_scratch_paths(root, scratch_parent):
    """Ownership comes from this run's append log, never a directory glob."""
    import json
    parent = Path(scratch_parent).resolve()
    paths = set()
    with (Path(root)/'workload-units.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('event') != 'SCRATCH_CREATED':
                continue
            path = Path(row['path'])
            if path.is_symlink() or path.parent.resolve() != parent or not path.name.startswith('sevc-scoped-'):
                raise ValueError('scratch ownership/path mismatch')
            paths.add(path)
    return paths


def finish_scratch_cleanup(study):
    """Normal completion only, after all group audits; never run on failure."""
    paths = registered_scratch_paths(study.root, study.config['scratch_parent'])
    for path in sorted(paths):
        release_scratch(study, path)
    remaining = [str(p) for p in paths if p.exists()]
    study.events({'event':'FINAL_SCRATCH_CLEANUP', 'owned_directories':len(paths),
                  'remaining':remaining, 'passed':not remaining})
    if remaining:
        raise RuntimeError('completed experiment retains owned scratch: '+str(remaining))
    return len(paths)


def assert_no_open_handles(source):
    """Linux maintenance gate; never evict a process to acquire ownership."""
    source = source.resolve()
    proc=Path('/proc')
    if not proc.exists():raise RuntimeError('offline handle gate requires Linux /proc')
    for process in proc.iterdir():
        if not process.name.isdigit():continue
        try:
            for fd in (process/'fd').iterdir():
                try:target=Path(os.readlink(fd))
                except FileNotFoundError:continue
                if target.is_absolute() and target.is_relative_to(source):
                    raise RuntimeError('scratch has open process handle: '+process.name)
            maps=(process/'maps').read_text()
            if str(source)+'/' in maps:
                raise RuntimeError('scratch has mapped process storage: '+process.name)
        except (FileNotFoundError,ProcessLookupError):continue
