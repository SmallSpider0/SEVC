"""Release only sealed trajectory tensors after their last registered use."""
from pathlib import Path
from sevc.core.artifacts import sha256_file


def release_trajectory_tensors(folder, emit):
    folder=Path(folder)
    paths=sorted(folder.glob('step-*.pt'))
    if any(path.is_symlink() for path in paths):
        raise ValueError('trajectory cleanup cannot follow links')
    seal=[{'path':str(path),'sha256':sha256_file(path),'bytes':path.stat().st_size} for path in paths]
    emit({'event':'TRAJECTORY_HASH_SEAL_AFTER_LAST_USE','folder':str(folder),'files':seal})
    for row in seal:
        Path(row['path']).unlink()
    return sum(row['bytes'] for row in seal)
