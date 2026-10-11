"""CPU rejection tests for explicit shared8 admission; no real GPU claims."""
import copy,json,tempfile,unittest
from pathlib import Path
from reproduction.common8.shared_slurm import AUTHORIZATION,POLICY,authorization,validate,physical_indices
class SharedSlurmTests(unittest.TestCase):
    def fixture(self,root):
        base=Path(root).resolve();(base/'preparation').mkdir()
        a=dict(authorization=AUTHORIZATION,base=str(base),physical_indices=list(range(8)),
               physical_gpu_uuids=['GPU-fixture-'+str(i) for i in range(8)],training_world_size=8,
               actor_micro_batch_per_rank=2,vllm_gpu_memory_utilization=.1,cpu_request=96,memory_request='512G',
               preserve_foreign_jobs=True,source_completed_outer=5,
               allowed_native_jobs={'3371':dict(JobId='3371',NodeList='h100-8')})
        path=base/'preparation/shared-slurm-authorization.json';path.write_text(json.dumps(a))
        _,sha=authorization(path,base)
        run=dict(slurm_gpu_sharing_policy=POLICY,checkpoint_dir=str(base/'checkpoints/search-g1-common8'),
                 slurm_shared_authorization_path=str(path),slurm_shared_authorization_sha256=sha)
        job=dict(JobId='4000',JobState='RUNNING',NodeList='h100-8',NumNodes='1',NumCPUs='96',MinMemoryNode='512G',OverSubscribe='YES')
        return run,job,[[str(i),u,'80000'] for i,u in enumerate(a['physical_gpu_uuids'])],a['physical_gpu_uuids']
    def test_actual_cpu_allocation_and_all_eight_shared_uuid_mapping(self):
        with tempfile.TemporaryDirectory() as root:
            run,job,gpus,uuids=self.fixture(root)
            self.assertTrue(validate(job,'h100-8',list(map(str,range(8))),['3371','4000'],gpus,[['GPU-fixture-0','123']],uuids,[['GPU-fixture-0','123']],run))
            allocation=dict(slurm_gpu_sharing_policy=POLICY,actual_allocated_indices=[],slurm_reserved_gpu_count=0,
                            actual_shared_physical_indices=list(range(8)),actual_cuda_runtime_uuids=uuids)
            self.assertEqual(physical_indices(allocation),list(range(8)))
            allocation['actual_allocated_indices']=list(range(8))
            with self.assertRaises(AssertionError):physical_indices(allocation)
    def test_budget_device_and_unknown_job_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run,job,gpus,uuids=self.fixture(root)
            for key,value in [('NumCPUs','48'),('MinMemoryNode','256G'),('JobState','PENDING'),('NodeList','h100-10'),('TresPerNode','gres:gpu:h100:8')]:
                bad=dict(job,**{key:value})
                with self.assertRaises(AssertionError):validate(bad,'h100-8',list(map(str,range(8))),['4000'],gpus,[],uuids,[],run)
            with self.assertRaises(AssertionError):validate(job,'h100-8',list(map(str,range(8))),['4000','9999'],gpus,[],uuids,[],run)
            with self.assertRaises(AssertionError):validate(job,'h100-8',list(map(str,range(8))),['4000'],gpus,[],list(reversed(uuids)),[],run)
            with self.assertRaises(AssertionError):validate(job,'h100-8',list(map(str,range(8))),['4000'],gpus,[['GPU-fixture-0','other']],uuids,[],run)
    def test_missing_authorization_and_legacy_indices(self):
        self.assertEqual(physical_indices({'actual_allocated_indices':[0,1,2,3,5,6]}),[0,1,2,3,5,6])
        with tempfile.TemporaryDirectory() as root:
            run,job,gpus,uuids=self.fixture(root)
            p=Path(run['slurm_shared_authorization_path']);a=json.loads(p.read_text());a['training_world_size']=4;p.write_text(json.dumps(a))
            with self.assertRaises(AssertionError):authorization(p,Path(root))
