import copy
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .pj_entry import BASE,ROOT,JOB,MODEL_FILES,VERSIONS,validate_spec,validate_scheduler,validate_submission_params,validate_gate
from .contract import recipe,JOB as OLD_JOB,UPSTREAM,MODEL_REVISION,MANIFEST_SHA


class Tests(unittest.TestCase):
    def setUp(self):
        self.script='''algorithm.adv_estimator=tree \\
actor_rollout_ref.rollout.tree_search=true \\
actor_rollout_ref.actor.optim.lr=$lr \\
trainer.total_training_steps=180 \\
max_turns=3 \\
'''
        self.plan=dict(protocol='official_treegrpo_pj_inspected_replacement_v1',job=JOB,
            root=ROOT+'/runs/fair-replacement-v3',upstream_source=ROOT+'/deployment-pj-v1/upstream',
            upstream_commit=UPSTREAM,model_revision=MODEL_REVISION,model=BASE+'/assets/model',
            model_files=MODEL_FILES,data=BASE+'/common-data',data_manifest_sha256=MANIFEST_SHA,
            required_versions=VERSIONS,allocation=dict(gpus=8,cpus=48,memory_gib=512,
                machine_type='n2ls.ie.i60b',gpu_model='A100',gpu_memory_gib=80),
            old_failure=dict(job_id='139034',state='FAILED',exit_code='1',complete_iterations=55,native_checkpoint_absent=True),
            actual_iterations_expected=179,no_auto_retry=True,retriever='http://shared:8000',
            intended_node=None,no_hostname_affinity=True,
            placement_recovery=dict(job='zhangjunming-ttt-treegrpo-pj-v1',
                uid='e078d102-57e4-4ed1-a359-610671e4e42b',
                status='TERMINATED_NEVER_STARTED_ZERO_GPU_EXECUTION',allocated_seconds=0,entry_absent=True),
            startup_recovery=dict(job='zhangjunming-ttt-treegrpo-pj-v2',
                uid='f51ffc58-4fdf-418d-914a-6b8c2220010f',
                status='FAILED_BEFORE_SCIENTIFIC_TRAINING_DNS_API_CHECK',
                scientific_updates=0,scientific_worker_not_started=True))
        self.plan['overrides']=[v.replace('='+OLD_JOB,'='+JOB) for v in recipe(self.script,
            self.plan['model'],self.plan['data'],self.plan['root'],self.plan['retriever'])]

    def test_frozen_science_and_allocation(self):
        self.assertEqual(validate_spec(self.plan,self.script),Path(self.plan['root']))

    def test_scientific_change_rejected(self):
        p=copy.deepcopy(self.plan);p['overrides']=[s.replace('lr=1e-6','lr=2e-6') for s in p['overrides']]
        with self.assertRaises(AssertionError):validate_spec(p,self.script)

    def test_external_gate_requires_actual_uid_resources_and_plan(self):
        sub=dict(job=JOB,uid='uid',plan_sha256='sha')
        gate=dict(plan_sha256='sha',actual=dict(name=JOB,uid='uid',state='RUNNING',resources=self.plan['allocation']))
        validate_gate(self.plan,sub,gate)
        for altered in [dict(gate,plan_sha256='other'),dict(gate,actual=dict(gate['actual'],state='PENDING')),dict(gate,actual=dict(gate['actual'],uid='other'))]:
            with self.assertRaises(AssertionError):validate_gate(self.plan,sub,altered)

    def test_publisher_waits_running_publishes_once_and_cannot_replay(self):
        from . import pj_gate
        from .contract import sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=dict(self.plan,root=tmp,wrapper_commit='frozen-source')
            f=root/'plan.json';f.write_text(json.dumps(p));sub=dict(job=JOB,uid='actual-uid',plan_sha256=sha(f))
            (root/'submission.json').write_text(json.dumps(sub))
            resources=dict(accelerate_device_count=8,cpu_count=48,memory_gib=512,
                machine_types=['n2ls.ie.i60b'],accelerate_device_model='A100',accelerate_device_memory_gib=80)
            pending=dict(uid='actual-uid',name=JOB,status=dict(state='PENDING'))
            running=dict(uid='actual-uid',name=JOB,status=dict(state='RUNNING'),
                spec=dict(vc_job=dict(tasks=[dict(replicas=1,resource_spec=resources)])))
            client=SimpleNamespace(get_training_job=unittest.mock.Mock(side_effect=[pending,running]))
            module=SimpleNamespace(get_cluster=lambda:SimpleNamespace(client=client))
            spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None))
            with patch.object(pj_gate.importlib.util,'spec_from_file_location',return_value=spec),patch.object(pj_gate.importlib.util,'module_from_spec',return_value=module),patch.object(pj_gate.time,'sleep'):
                pj_gate.run(str(f))
                actual=json.loads((root/'scheduler-release-once.json').read_text())
                self.assertEqual(actual['actual']['uid'],'actual-uid')
                self.assertEqual(actual['actual']['state'],'RUNNING')
                self.assertFalse((root/'scheduler-release.tmp').exists())
                with self.assertRaises(FileExistsError):pj_gate.run(str(f))

    def test_no_missing_native_or_live_parent(self):
        for change in [dict(state='RUNNING'),dict(native_checkpoint_absent=False)]:
            p=copy.deepcopy(self.plan);p['old_failure'].update(change)
            with self.assertRaises(AssertionError):validate_spec(p,self.script)

    def test_no_smaller_gpu_or_other_initialization(self):
        for key,value in [('gpus',4),('gpu_memory_gib',40)]:
            p=copy.deepcopy(self.plan);p['allocation'][key]=value
            with self.assertRaises(AssertionError):validate_spec(p,self.script)
        p=copy.deepcopy(self.plan);p['model_revision']='other'
        with self.assertRaises(AssertionError):validate_spec(p,self.script)

    def test_real_uid_and_backend_no_synthetic_slurm(self):
        i=dict(job=JOB,plan_sha256='p');s=dict(i,uid='uid');a=dict(name=JOB,uid='uid',state='RUNNING',resources=self.plan['allocation'])
        with patch.dict(os.environ,dict(SEARCH_R1_EXECUTION_BACKEND='pj_pt',SEARCH_R1_JOB_ID=JOB),clear=True):
            validate_scheduler(self.plan,i,s,a)
            with self.assertRaises(AssertionError):validate_scheduler(self.plan,i,s,dict(a,uid='other'))
            os.environ['SLURM_JOB_ID']='made-up'
            with self.assertRaises(AssertionError):validate_scheduler(self.plan,i,s,a)

    def test_explicit_hostname_affinity_and_executed_prior_block(self):
        params=dict(job_name=JOB,num_nodes=1,gpus_per_task=8,cpus_per_task=48,
                    memory_per_task=512,retry=0,no_env=True)
        validate_submission_params(self.plan,params)
        with self.assertRaises(AssertionError):validate_submission_params(self.plan,dict(params,include='ecp-pcluster-c480'))
        p=copy.deepcopy(self.plan);p['placement_recovery']['allocated_seconds']=1
        with self.assertRaises(AssertionError):validate_spec(p,self.script)


if __name__=='__main__':unittest.main()
