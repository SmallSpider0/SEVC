"""Authenticated private mmap for immutable, run-owned disposable tensors.

This caches file authentication, never replay answers. Every access checks the
file identity. Protocol content commitments remain the consumer's responsibility.
"""
from pathlib import Path
import threading


class AuthenticatedTensorFile:
    def __init__(self, path, digest):
        self.path, self.digest = Path(path), digest
        self._fingerprint = None
        self._lock = threading.Lock()

    @classmethod
    def inspect_owned(cls, path):
        """Hash newly written owned bytes once, before any actor receives them."""
        from sevc.core.artifacts import sha256_file
        result = cls(path, None)
        before = result.fingerprint()
        result.digest = sha256_file(result.path)
        if before != result.fingerprint():
            raise ValueError("owned tensor changed during authentication")
        result._fingerprint = before
        return result

    def fingerprint(self):
        s = self.path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    @classmethod
    def inspect_trusted_owned(cls, path):
        """Disposable trusted scratch: metadata identity only, no content SHA."""
        result = cls(path, None)
        result._fingerprint = result.fingerprint()
        return result

    def load(self):
        import torch
        from sevc.core.artifacts import sha256_file
        with self._lock:
            before = self.fingerprint()
            if self._fingerprint is None:
                if sha256_file(self.path) != self.digest or self.fingerprint() != before:
                    raise ValueError("tensor file integrity mismatch")
                self._fingerprint = before
            elif self._fingerprint != before:
                raise ValueError("immutable tensor file identity changed")
            # MAP_PRIVATE is torch's default: tensor mutation cannot alter disk.
            result = torch.load(self.path, map_location="cpu", weights_only=False, mmap=True)
            if self.fingerprint() != before:
                raise ValueError("tensor file changed during mmap load")
            return result


class ResidentDeliveryBudget:
    """Reserve tensor payload bytes once per job, safe across owner lanes."""
    def __init__(self, limit):
        if int(limit) < 0:
            raise ValueError("negative resident delivery budget")
        self.limit, self.used = int(limit), 0
        self._lock = threading.Lock()

    def reserve(self, proof):
        states = (proof.initial_state, *proof.checkpoints,
                  proof.optimizer_initial_state or {}, *proof.optimizer_checkpoints)
        values = [v for state in states for v in state.values()]
        values.extend(v for batch in proof.batches for v in batch)
        size = sum(v.numel() * v.element_size() for v in values)
        with self._lock:
            if self.used + size > self.limit:
                return False
            self.used += size
            return True
