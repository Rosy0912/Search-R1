"""Bind diagnostic requests to the admitted private retriever process."""
import json
from pathlib import Path
from urllib.parse import urlparse, urlunparse

if __package__:
    from .common import file_sha, stat_identity
else:
    from common import file_sha, stat_identity

IDENTITY_KEYS = ("pid", "uid", "starttick", "source_identity_sha256")


def field(value, path):
    for key in path.split("."):
        value = value[key]
    return value


class RetrieverAdmission:
    def __init__(self, path, url, allow_fallback=False, request_health=None):
        self.path = Path(path).resolve()
        self.admission = json.loads(self.path.read_text())
        endpoint = urlparse(url)
        if endpoint.port not in (18005, 18004) or endpoint.path != "/retrieve":
            raise ValueError("Unapproved diagnostic retriever endpoint")
        if endpoint.port == 18004 and not allow_fallback:
            raise ValueError("18004 fallback requires an explicit argument and its own admission")
        if self.admission.get("accepted") is not True or self.admission.get("retriever_url") != url:
            raise ValueError("Retrieval admission does not accept the exact endpoint")
        self.expected = {key: self.admission[key] for key in IDENTITY_KEYS}
        self.paths = self.admission.get("health_identity_paths", {key: key for key in IDENTITY_KEYS})
        if set(self.paths) != set(IDENTITY_KEYS):
            raise ValueError("Health identity paths omit required admitted process/source fields")
        self.gate_path = Path(self.admission["full_gpu_gate_path"]).resolve()
        before = stat_identity(self.gate_path)
        if file_sha(self.gate_path) != self.admission["full_gpu_gate_sha256"]:
            raise ValueError("Full GPU retriever gate content hash differs")
        gate = json.loads(self.gate_path.read_text())
        if gate.get("accepted") is not True:
            raise ValueError("Full GPU retriever gate is not accepted")
        candidate = gate["candidate_health"]
        # A full GPU gate is measured before admit, when ready=False is normal.
        # It binds the tested process/source and GPU readiness, not admission.
        self.check_identity(candidate, require_ready=False)
        if stat_identity(self.gate_path) != before:
            raise ValueError("Full GPU gate changed during startup read")
        self.gate_stat = before
        self.admission_stat = stat_identity(self.path)
        self.identity = dict(path=str(self.path), sha256=file_sha(self.path), admitted=self.expected,
                             full_gpu_gate_path=str(self.gate_path), full_gpu_gate_sha256=self.admission["full_gpu_gate_sha256"],
                             endpoint=url, explicit_18004_fallback=endpoint.port == 18004,
                             health_identity_paths=self.paths)
        self.health_url = urlunparse(endpoint._replace(path="/health", query="", fragment=""))
        self.request_health = request_health or self._request_health

    def check_identity(self, health, require_ready=True):
        if require_ready and health.get("ready") is not True:
            raise RuntimeError("Admitted diagnostic retriever is not ready")
        gpu_path = self.admission.get("health_gpu_ready_path", "gpu_ready")
        if field(health, gpu_path) is not True:
            raise RuntimeError("Diagnostic retriever GPU gate/readiness is not active")
        for key, expected in self.expected.items():
            if field(health, self.paths[key]) != expected:
                raise RuntimeError("Admitted retriever process/source identity changed: " + key)

    def _request_health(self):
        import requests
        reply = requests.get(self.health_url, timeout=10)
        reply.raise_for_status()
        return reply.json()

    def check_live(self):
        if stat_identity(self.path) != self.admission_stat or stat_identity(self.gate_path) != self.gate_stat:
            raise RuntimeError("Diagnostic retriever admission/gate metadata changed")
        health = self.request_health()
        self.check_identity(health)
        return health
