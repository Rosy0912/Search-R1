import json
from pathlib import Path
import tempfile
import unittest
from .contracts import communication_transport,sha


class TransportTests(unittest.TestCase):
    def test_default_remains_unmodified(self):
        self.assertEqual(communication_transport({}),{})

    def test_only_completed_eight_rank_probe_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'probe.json'
            flags={'NCCL_P2P_DISABLE':'1','NCCL_IB_DISABLE':'1'}
            proof=dict(status='EIGHT_RANK_COMMUNICATION_PROBE_PASSED',transport=flags,
                       ranks=[dict(rank=r,cases=[dict(mib=m,broadcast_and_reduce_values_verified=True)
                                                for m in (1,64,256)]) for r in range(8)])
            p.write_text(json.dumps(proof))
            plan=dict(communication_transport=flags,communication_probe=str(p),communication_probe_sha256=sha(p))
            self.assertEqual(communication_transport(plan),flags)
            proof['ranks'].pop();p.write_text(json.dumps(proof));plan['communication_probe_sha256']=sha(p)
            with self.assertRaises(ValueError):communication_transport(plan)

    def test_arbitrary_environment_override_rejected(self):
        with self.assertRaises(ValueError):
            communication_transport({'communication_transport':{'NCCL_P2P_DISABLE':'0'}})
