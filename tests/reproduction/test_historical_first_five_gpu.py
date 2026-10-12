"""Tiny actual scalar sample files; current process/GPU liveness is never mocked in."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reproduction.common8.gpu_training_admission import _coverage, verify_historical_training_gpu_samples
from reproduction.common8.policy_calibration_plan import canonical_bytes


def fixture(root):
    base=Path(root).resolve();(base/'logs').mkdir()
    run=dict(run_id='a'*32,execution_id='b'*32,execution_backend='ssh_local',profile='common8',started_unix=1,
        source_sha256={'scorer.py':'c'*64},training_world_size=4,
        training_physical_indices=[0,1,2,3],auxiliary_physical_indices=[5,6])
    claim=dict(run_id=run['run_id'],execution_id=run['execution_id'],execution_backend='ssh_local',
        slurm_job_id=None,hostname='fixture-host',claimed_unix=5)
    physical=[0,1,2,3,5,6];inventory=[[str(i),'fixture-gpu-'+str(i),'81920'] for i in range(8)]
    allocation=dict(accepted=True,run_id=run['run_id'],execution_id=run['execution_id'],execution_backend='ssh_local',
        slurm_job_id=None,node='fixture-host',actual_allocated_indices=physical,training_physical_indices=[0,1,2,3],
        physical_gpu_inventory=inventory,actual_cuda_runtime_uuids=['fixture-gpu-'+str(i) for i in range(4)],
        assigned_gpu_uuids={str(i):'fixture-gpu-'+str(i) for i in physical})
    schedulers={str(rank):dict(selected_rows=[row for step in range(1,6) for row in (
        dict(logical_update=step,kind='BEGIN',observed_unix=4+2*step),
        dict(logical_update=step,kind='COMPLETE',observed_unix=4.5+2*step))]) for rank in range(4)}
    rows=[]
    for end in (4.25,6.25,8.25,10.25,12.25,14.25,20.25):
        rows.append(dict(schema_version=1,role='actual_common8_gpu_occupancy',run_id=run['run_id'],
            execution_id=run['execution_id'],execution_backend='ssh_local',slurm_job_id=None,hostname='fixture-host',
            source_map_sha256=hashlib.sha256(canonical_bytes(run['source_sha256'])).hexdigest(),
            sampler_identity=dict(pid=101,start_ticks=102,sid=101,uid=0),sampling_interval_seconds=2,
            query_started_unix=end-.25,unix=end,gpus=[dict(index=i,uuid='fixture-gpu-'+str(i),
                used_mib=100+i,total_mib=81920,util_percent=50) for i in physical]))
    raw=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
    path=base/'logs'/('gpu-samples-'+run['run_id']+'.jsonl');path.write_bytes(raw)
    (base/'logs/training-launch-claim.json').write_text(json.dumps(claim)+'\n')
    allocation_path=base/'logs/native-allocation-admission.json';allocation_path.write_text(json.dumps(allocation)+'\n')
    saved=_coverage(rows,run,claim,allocation,schedulers,entered=20)
    saved.update(receipt_path=str(path),observed_complete_prefix_sha256=hashlib.sha256(raw).hexdigest(),
        allocation_receipt_sha256=hashlib.sha256(allocation_path.read_bytes()).hexdigest(),
        semantics=saved['semantics']+' A newly acquired sample after gate entry and live sampler/trainer births are mandatory.')
    return base,run,schedulers,rows,path,saved


class HistoricalGPUFirstFiveTests(unittest.TestCase):
    def test_old_births_are_not_claimed_live_and_later_rows_do_not_change_original_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,schedulers,rows,path,saved=fixture(root)
            proof=verify_historical_training_gpu_samples(base,run,schedulers,saved)
            self.assertFalse(proof['current_liveness_claimed']);self.assertEqual(proof['original_gate']['sample_count'],7)
            later=dict(rows[-1],query_started_unix=22,unix=22.25)
            with path.open('ab') as stream:stream.write(json.dumps(later).encode()+b'\n'+b'{"partial":')
            proof2=verify_historical_training_gpu_samples(base,run,schedulers,saved)
            self.assertEqual(proof['original_gate'],proof2['original_gate'])

    def test_raw_prefix_device_bound_fields_and_allocation_hash_mutations_rejected(self):
        for kind in ('raw','peak','gap','last_sample','allocation','receipt_path'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                base,run,schedulers,rows,path,saved=fixture(root)
                if kind=='raw':path.write_bytes(path.read_bytes().replace(b'"used_mib": 100',b'"used_mib": 101',1))
                elif kind=='peak':saved['sampled_peak_mib']['0']+=1
                elif kind=='gap':saved['maximum_sampling_gap_seconds']+=1
                elif kind=='last_sample':saved['last_sample_unix']=30
                elif kind=='allocation':saved['allocation_receipt_sha256']='0'*64
                else:saved['receipt_path']='/elsewhere/ledger'
                with self.assertRaises(AssertionError):verify_historical_training_gpu_samples(base,run,schedulers,saved)

    def test_same_timestamps_cannot_replace_missing_original_gate_sample(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,schedulers,rows,path,saved=fixture(root)
            raw=path.read_bytes();last=raw.rfind(b'\n',0,len(raw)-1)+1
            path.write_bytes(raw[:last])
            with self.assertRaisesRegex(AssertionError,'sample is missing'):
                verify_historical_training_gpu_samples(base,run,schedulers,saved)


if __name__=='__main__':unittest.main()
