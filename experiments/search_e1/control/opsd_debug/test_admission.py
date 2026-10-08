"""CPU admission tests including the real pre-admit ready=False gate pattern."""
import json
from pathlib import Path
import tempfile
import unittest

if __package__:
    from .admission import RetrieverAdmission
    from .common import file_sha
else:
    from admission import RetrieverAdmission
    from common import file_sha


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.health = dict(pid=123, uid=1005, starttick=456, source_identity_sha256="a" * 64, ready=True, gpu_ready=True)
        historical = dict(self.health, ready=False)
        self.gate = self.root / "gate.json"
        self.gate.write_text(json.dumps(dict(accepted=True, candidate_health=historical)))
        self.admission = self.root / "admission.json"
        self.spec = dict(accepted=True, retriever_url="http://10.120.17.101:18005/retrieve",
                         full_gpu_gate_path=str(self.gate), full_gpu_gate_sha256=file_sha(self.gate),
                         **{key: self.health[key] for key in ("pid", "uid", "starttick", "source_identity_sha256")})
        self.admission.write_text(json.dumps(self.spec))

    def tearDown(self):
        self.temporary.cleanup()

    def build(self):
        return RetrieverAdmission(self.admission, self.spec["retriever_url"], request_health=lambda: self.health)

    def test_pre_admit_gate_can_be_bound_then_live_ready_required(self):
        guard = self.build()
        self.assertEqual(guard.check_live()["pid"], 123)
        self.health["ready"] = False
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            guard.check_live()

    def test_process_source_and_gpu_identity_changes_rejected(self):
        guard = self.build()
        for key in ("pid", "uid", "starttick", "source_identity_sha256", "gpu_ready"):
            original = self.health[key]
            self.health[key] = False if key == "gpu_ready" else "changed"
            with self.assertRaises(RuntimeError):
                guard.check_live()
            self.health[key] = original

    def test_gate_change_rejected_without_rehashing_each_case(self):
        guard = self.build()
        self.gate.write_text(self.gate.read_text() + " ")
        with self.assertRaisesRegex(RuntimeError, "metadata changed"):
            guard.check_live()

    def test_full_gpu_hash_mismatch_rejected(self):
        self.spec["full_gpu_gate_sha256"] = "b" * 64
        self.admission.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.build()

    def test_historical_gpu_not_ready_is_not_a_valid_gate(self):
        self.gate.write_text(json.dumps(dict(accepted=True, candidate_health=dict(self.health, ready=False, gpu_ready=False))))
        self.spec["full_gpu_gate_sha256"] = file_sha(self.gate)
        self.admission.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(RuntimeError, "GPU gate/readiness"):
            self.build()

    def test_fallback_is_explicit_and_has_its_own_url_admission(self):
        with self.assertRaises(ValueError):
            RetrieverAdmission(self.admission, "http://10.120.17.101:18004/retrieve", request_health=lambda: self.health)
        with self.assertRaises(ValueError):
            RetrieverAdmission(self.admission, "http://10.120.17.101:18004/retrieve", allow_fallback=True, request_health=lambda: self.health)


if __name__ == "__main__":
    unittest.main()
