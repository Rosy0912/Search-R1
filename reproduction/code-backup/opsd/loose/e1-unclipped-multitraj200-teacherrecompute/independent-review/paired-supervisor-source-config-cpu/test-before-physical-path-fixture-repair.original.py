"""Small local software interfaces only; no production main or future approval."""
import copy
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

H = Path(__file__).resolve().parents[2]
C = H / 'final200-paired-supervisor-candidate'
REPO = H.parent / 'opsd-final200-paired-eval-source'
sys.path.insert(0, str(REPO))
original_get_code = importlib.machinery.SourceFileLoader.get_code
def exact_get_code(loader, fullname):
    path = Path(loader.path)
    if path.is_relative_to(C) or path.is_relative_to(REPO):
        return compile(path.read_bytes(), str(path), 'exec', dont_inherit=True, optimize=0)
    return original_get_code(loader, fullname)
importlib.machinery.SourceFileLoader.get_code = exact_get_code

def load(path, name):
    module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    return module
s = load(C/'paired_supervisor.py', 'independent_paired_supervisor')
cpu = load(C/'paired_cpu_prepare.py', 'independent_paired_cpu')
contract = load(C/'paired_eval_contract.py', 'independent_paired_contract')
author = load(C/'test_paired_supervisor.py', 'independent_selected_author_cases')

class TestInterfaces(unittest.TestCase):
    def test_actual_frozen_config_no_role_or_training_budget_drift(self):
        result=unittest.TestResult()
        author.TestControl('test_actual_frozen_c5bf_full_config_and_proposed_stage_no_review_cycle').run(result)
        self.assertTrue(result.wasSuccessful(),result.errors+result.failures)
        q = dict(common_manifest=dict(path=str(s.ROOT/'manifest.json'), sha256=contract.COMMON))
        input_record = dict(path=str(s.ROOT/'future/inputs.json'), sha256='d'*64)
        with patch.object(s, 'REPO', REPO):
            config = cpu.compose_role(s, q, s.ROOT/'paired-final200-runs/eval-901', 'actual_hf200', input_record, contract)
        for fields, bad in [(('trainer','total_training_steps'),True), (('actor_rollout_ref','rollout','n_agent'),5),
                (('baselines','opsd_e1','profile'),'multitraj200'), (('algorithm','kl_ctrl','kl_coef'),.001),
                (('actor_rollout_ref','actor','entropy_coeff'),.1), (('data','val_batch_size'),False)]:
            wrong=copy.deepcopy(config); target=wrong
            for key in fields[:-1]: target=target[key]
            target[fields[-1]]=bad
            with self.assertRaises(ValueError): contract.validate_resolved_config(wrong)
        self.assertEqual(config['actor_rollout_ref']['model']['path'], str(s.HF200))

    def test_models_actual_hf_training_state_and_service_metadata_interfaces(self):
        for name in ('test_complete_models_real_training_state_directory_and_unknown_entries',
                     'test_service_observation_view_no_added_forward_and_gpu_devices_strict'):
            result=unittest.TestResult(); author.TestControl(name).run(result)
            self.assertTrue(result.wasSuccessful(), result.errors+result.failures)

    def test_actual_signed_first5_valid_signature_cannot_upgrade_to200(self):
        public_raw=(H/'job3304-first5-complete-originals-v3/raw/bindings/public.json').read_bytes()
        self.assertEqual(hashlib.sha256(public_raw).hexdigest(),s.PUBLIC_SHA)
        envelope_raw=(H/'first5-actual-root-release-v1/signed-envelope.json').read_bytes()
        envelope=json.loads(envelope_raw); public=json.loads(public_raw)
        gate=load(REPO/'search_r1/baselines/checkpoint_gate.py','independent_frozen_signature_gate')
        self.assertEqual(hashlib.sha256((REPO/'search_r1/baselines/checkpoint_gate.py').read_bytes()).hexdigest(),s.GATE_SHA)
        gate.verify_rsa_signature(envelope['payload'],envelope['signature_hex'],public)
        def ref(value,**kw):return (public if value['path']==str(s.PUBLIC) else envelope), {}
        with patch.object(s,'reference',side_effect=ref),patch.object(s,'exact_module',return_value=(gate,{})):
            with self.assertRaisesRegex(ValueError,'Execution request schema differs'):
                s.load_request(Path('/fixture/first5.json'),'a'*64)
        altered=copy.deepcopy(envelope['payload']);altered['outer_step']=200
        with self.assertRaises(ValueError):gate.verify_rsa_signature(altered,envelope['signature_hex'],public)

    def test_signed_ready_fields_fail_closed_independently_of_crypto_fixture(self):
        # Gate here is an explicit schema fixture; actual RSA is exercised above.
        ready=dict(request={'sha256':'a'*64},nonce='b'*64,parent_identity={'pid':17,'start_ticks':19},
            gpu_uuids=['GPU-'+str(i) for i in range(8)],leases=[{'fd':i+10} for i in range(9)],
            stages={r:{'sha256':('c' if r=='official_base' else 'd')*64} for r in s.ROLES},
            configurations={r:{'sha256':('e' if r=='official_base' else 'f')*64} for r in s.ROLES})
        expected=dict(version=1,kind='ROOT_SIGNED_PAIRED_FINAL200_PRE_GPU_APPROVAL_INITIALIZATION_ONLY',
            ready_sha256='1'*64,request_sha256='a'*64,nonce='b'*64,parent_identity=ready['parent_identity'],
            gpu_uuids=ready['gpu_uuids'],leases=ready['leases'],stage_sha256={r:ready['stages'][r]['sha256'] for r in s.ROLES},
            configuration_sha256={r:ready['configurations'][r]['sha256'] for r in s.ROLES},
            root_review={'path':'/fixture/root.json','sha256':'2'*64},peer_review={'path':'/fixture/peer.json','sha256':'3'*64})
        gate=types.SimpleNamespace(verify_rsa_signature=lambda *args:None)
        lease=types.SimpleNamespace(check=lambda:None)
        sampler=types.SimpleNamespace(sample=lambda:{'gpu_processes':[],'gpu_inventory':[{'used_mib':0,'total_mib':1,'utilization':0}]})
        def refs(value,**kw):
            role='ROOT_REVIEWER' if 'root' in value['path'] else 'INDEPENDENT_REVIEWER'
            return dict(version=1,role=role,status='PASS_ACTUAL_PAIRED_FINAL200_PRE_GPU_INITIALIZATION_ONLY',ready_sha256='1'*64,request_sha256='a'*64),{}
        with tempfile.TemporaryDirectory() as name:
            run=Path(name).resolve();(run/'pre-gpu-approval.json').write_bytes(b'fixture')
            def invoke(payload):
                raw=s.canonical({'payload':payload,'signature_hex':'fixture-no-authentication'})
                with patch.object(s,'capture',return_value=(raw,{'sha256':'4'*64})),patch.object(s,'reference',side_effect=refs),patch.object(s,'same_process',return_value=True):
                    return s.await_approval(run,ready,{'sha256':'1'*64},gate,1,lease,sampler)
            self.assertEqual(invoke(expected),{'sha256':'4'*64})
            for key,bad in [('version',True),('nonce','wrong'),('ready_sha256','0'*64),('parent_identity',{'pid':17,'start_ticks':20}),('leases',[]),('stage_sha256',ready['configurations']),('configuration_sha256',ready['stages'])]:
                wrong=copy.deepcopy(expected);wrong[key]=bad
                with self.assertRaises(ValueError):invoke(wrong)

    def test_adapter_hides_then_restores_only_assigned_visibility_before_main(self):
        ready={'parent_identity':{},'stages':{r:{'sha256':'c'*64} for r in s.ROLES}}
        consumed={'ready':{'sha256':'a'*64},'approval':{'sha256':'b'*64}}
        def ref(value,**kw):
            if value['path'].endswith('actual-approval-consumed.json'):return consumed,{}
            if value['path'].endswith('pre-gpu-ready.json'):return ready,{'sha256':'a'*64}
            if value['path'].endswith('pre-gpu-approval.json'):return {'payload':{'ready_sha256':'a'*64},'signature_hex':'fixture'},{}
            return {},{}
        gate=types.SimpleNamespace(verify_rsa_signature=lambda *args:None)
        def module(path,pin,name):return (gate if 'gate' in name else contract),{}
        calls=[]
        def native(*args):
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'');calls.append('native-hidden');return {},None,None
        def main(*args,**kw):
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0,1,2,3,4,5,6,7');calls.append('frozen-main-visible')
            self.assertIn('--config-name',sys.argv);self.assertIn('ppo_opsd_final200_eval',sys.argv)
            self.assertTrue(all(v in sys.argv for v in contract.inactive_overrides()))
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'','PAIRED_EVAL_ASSIGNED_CUDA_DEVICES':'0,1,2,3,4,5,6,7','OPSD_FINAL200_EVAL_STAGE_SHA256':'c'*64}),patch.object(s,'reference',side_effect=ref),patch.object(s,'capture',return_value=(None,{'sha256':'c'*64})),patch.object(s,'same_process',return_value=True),patch.object(s,'exact_module',side_effect=module),patch.object(cpu,'native_context',side_effect=native),patch.object(cpu.runpy,'run_module',side_effect=main):
            cpu.evaluate(s,{},s.ROOT/'fixture','actual_hf200')
        self.assertEqual(calls,['native-hidden','frozen-main-visible'])

    def test_frozen_configure_known_premain_data_mutation(self):
        from omegaconf import OmegaConf
        from search_r1.baselines import final200_evaluation as evaluation
        q=dict(common_manifest=dict(path=str(s.ROOT/'manifest.json'),sha256=contract.COMMON))
        with patch.object(s,'REPO',REPO):
            resolved=cpu.compose_role(s,q,s.ROOT/'paired-final200-runs/eval-901','official_base',dict(path='/fixture/inputs.json',sha256='d'*64),contract)
        cfg=OmegaConf.create(resolved)
        final=[str(Path(v).resolve()) for v in cfg.data.final_val_files]
        premain=list(cfg.data.val_files);self.assertNotEqual(premain,final)
        with patch.object(evaluation,'load_inputs',return_value={'inputs':[{'path':v} for v in final]}),patch.object(evaluation,'load_stage',return_value={'generation_budget':contract.BUDGET}):
            evaluation.configure(cfg,{'content_identity':{'sha256':contract.COMMON}})
        self.assertEqual(list(cfg.data.val_files),final);self.assertIs(cfg.data.return_raw_chat,True)
        self.assertEqual(cfg.trainer.total_training_steps,0)

if __name__=='__main__':unittest.main(verbosity=2)
