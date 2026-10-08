"""CPU-only exact gradient partition and full optimizer/RNG-resume fixtures."""
import copy
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import torch
from trainer import (RNG_NAMES, actual_adam_state, batches, capture_rng,
    optimizer_tensor_digest, rank_batch, restore_rng, save_torch, state_to_cpu,
    tensor_digest, validate_science, verify_rank_state, verify_resume_cursor,
    verify_sampling_identity, verify_committed_receipts, validate_rng, verify_source100, verify_source_metadata, atomic_checkpoint_directory, use_frozen_runtime, runtime_rank_identity)


def config():
    return dict(_config_sha256='c'*64,source_model_manifest=dict(sha256='s'*64),
        runtime_manifest=dict(sha256='r'*64),logging=dict(topk=32,topk_max_tokens_per_outer=0),
        scientific=dict(loss_reduction='trajectory_mean',learning_rate=5e-6,micro_batch=1,
        global_batch=32,lora_rank=16,lora_alpha=32,grad_norm_clip=1,chunk=32,max_context=8192,
        seed=42,warmup_updates=0,component_clip=None,outer_iterations=10,
        optimizer=dict(kind='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01)))


def adam(parameter):
    return torch.optim.AdamW([parameter],lr=5e-6,betas=(.9,.999),eps=1e-8,weight_decay=.01)


class TrainerTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(42); torch.set_num_threads(1)
    def test_runtime_uuid_torch24_properties_without_uuid_uses_actual_mapping(self):
        # The production Torch2.4 device properties have no uuid. This fixture
        # intentionally has only name/total_memory and never initializes CUDA.
        cuda=SimpleNamespace(get_device_properties=Mock(return_value=SimpleNamespace(name='RTX',total_memory=24<<30)),
            mem_get_info=Mock(return_value=(3<<30,24<<30)),
            max_memory_allocated=Mock(return_value=12<<30),max_memory_reserved=Mock(return_value=14<<30))
        fake_torch=SimpleNamespace(cuda=cuda)
        owner=dict(owner=dict(physical_gpus=list(range(8)),gpu_uuids=['GPU-fixture-'+str(i) for i in range(8)]))
        inventory='\n'.join(str(i)+', GPU-fixture-'+str(i) for i in range(8))
        proc_stat='1234 (fixture) '+' '.join(['R','1']+['0']*17+['678'])
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}),patch('subprocess.check_output',return_value=inventory) as query,patch('pathlib.Path.read_text',return_value=proc_stat):
            result=runtime_rank_identity(fake_torch,SimpleNamespace(index=2),2,owner)
        self.assertEqual((result['physical_gpu'],result['gpu_uuid'],result['local_rank']),(2,'GPU-fixture-2',2))
        self.assertEqual(result['actual_free_bytes'],3<<30)
        self.assertEqual(result['starttick'],678)
        cuda.get_device_properties.assert_not_called()
        query.assert_called_once_with(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)

    def test_runtime_uuid_mapping_and_actual_inventory_mismatch_rejected(self):
        cuda=SimpleNamespace(mem_get_info=Mock(),max_memory_allocated=Mock(),max_memory_reserved=Mock())
        fake_torch=SimpleNamespace(cuda=cuda)
        owner=dict(owner=dict(physical_gpus=list(range(8)),gpu_uuids=['GPU-fixture-'+str(i) for i in range(8)]))
        inventory='\n'.join(str(i)+', GPU-fixture-'+str(i) for i in range(8))
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1,0,2,3,4,5,6,7'}),patch('subprocess.check_output',return_value=inventory) as query:
            with self.assertRaisesRegex(ValueError,'authorized physical GPU mapping'):
                runtime_rank_identity(fake_torch,SimpleNamespace(index=2),2,owner)
            query.assert_not_called()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}),patch('subprocess.check_output',return_value=inventory.replace('GPU-fixture-2','GPU-foreign')):
            with self.assertRaisesRegex(ValueError,'authorized eight-GPU inventory'):
                runtime_rank_identity(fake_torch,SimpleNamespace(index=2),2,owner)
        cuda.mem_get_info.assert_not_called()

    def test_pending_human_choices_fail_closed(self):
        c=config(); validate_science(c)
        for key,value in [('component_clip','UNRESOLVED'),('outer_iterations',None),('micro_batch',4),('loss_reduction','token_mean')]:
            bad=copy.deepcopy(c);bad['scientific'][key]=value
            with self.assertRaises(ValueError):validate_science(bad)

    def test_tail_8rank_micro_sum_exact_gradient(self):
        # Per-trajectory gradients are mean over its tokens; ranks SUM and real B.
        for size in [1,2,7,8,9,31,32]:
            parameter=torch.tensor(.4,requires_grad=True)
            lengths=[(i%5)+1 for i in range(size)]
            oracle=sum(((parameter-torch.arange(n,dtype=torch.float32)/7)**2).mean() for n in lengths)/size
            oracle.backward(); desired=parameter.grad.clone()
            rank_gradients=[]
            for rank in range(8):
                p=torch.tensor(.4,requires_grad=True)
                for i in rank_batch(list(range(size)),rank,8):
                    (((p-torch.arange(lengths[i],dtype=torch.float32)/7)**2).mean()/size).backward()
                rank_gradients.append(torch.zeros_like(p) if p.grad is None else p.grad)
            torch.testing.assert_close(sum(rank_gradients),desired,rtol=1e-6,atol=1e-7)

    def test_empty_round_zero_updates_and_tail_no_drop(self):
        self.assertEqual(batches([]),[])
        for size in [0,1,33,65,512]:
            bs=batches(list(range(size)))
            self.assertEqual(len(bs),math.ceil(size/32))
            self.assertEqual([x for b in bs for x in b],list(range(size)))
        self.assertEqual(batches(list(range(65)),32),[list(range(32,64)),[64]])
        with self.assertRaises(ValueError):batches(list(range(65)),31)

    def test_resume_cursor_has_same_data_and_no_repeated_committed_batch(self):
        state=dict(cursor=dict(outer_index=1,next_pair_cursor=32,usable_pair_count=65,order_sha256='o',encoded_sha256='e'))
        self.assertEqual(verify_resume_cursor(state,1,'o','e',65),32)
        self.assertEqual([v for b in batches(list(range(65)),32) for v in b],list(range(32,65)))
        for args in [(2,'o','e',65),(1,'bad','e',65),(1,'o','e',64)]:
            with self.assertRaises(ValueError):verify_resume_cursor(state,*args)

    def test_tensor_digest_scalar_moments_and_groups_observable(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=adam(p)
        (p**2).sum().backward();opt.step()
        state=state_to_cpu(opt.state_dict());digest=optimizer_tensor_digest(state)
        self.assertEqual(actual_adam_state(state,1,1),[1.])
        changed=copy.deepcopy(state);changed['state'][0]['exp_avg'][0]+=1
        self.assertNotEqual(digest,optimizer_tensor_digest(changed))
        changed=copy.deepcopy(state);changed['param_groups'][0]['lr']=1e-6
        self.assertNotEqual(digest,optimizer_tensor_digest(changed))
        self.assertEqual(tensor_digest({'step':torch.tensor(1.)}),tensor_digest({'step':torch.tensor(1.)}))

    def test_adam_saved_all_parameters_step_finite_not_count_only(self):
        p=torch.nn.Parameter(torch.tensor([1.]));opt=adam(p)
        self.assertEqual(actual_adam_state(opt.state_dict(),0,1),[])
        p.sum().backward();opt.step();state=state_to_cpu(opt.state_dict())
        for mutation in ['missing','step','nan','shape']:
            bad=copy.deepcopy(state)
            if mutation=='missing':bad['state']={}
            elif mutation=='step':bad['state'][0]['step']=torch.tensor(2.)
            elif mutation=='nan':bad['state'][0]['exp_avg'][0]=float('nan')
            else:bad['state'][0]['exp_avg_sq']=torch.ones(2)
            with self.assertRaises((ValueError,FloatingPointError)):actual_adam_state(bad,1,1)

    def test_complete_cpu_adam_rng_resume_matches_uninterrupted(self):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        p=torch.nn.Parameter(torch.tensor([.1,.2]));opt=adam(p)
        def step(p,opt):
            x=torch.tensor([random.random()+np.random.rand(),torch.rand(()).item()])
            opt.zero_grad();((p-x)**2).sum().backward();opt.step()
        with patch('torch.cuda.get_rng_state_all',return_value=[]),patch('torch.cuda.set_rng_state_all'):
            step(p,opt);rng=capture_rng(torch,np)
            state=state_to_cpu(opt.state_dict());saved=p.detach().clone()
            with tempfile.TemporaryDirectory() as temp:
                file=Path(temp)/'state.pt';save_torch(file,dict(optimizer=state,rng=rng,adapter=saved),torch)
                step(p,opt);expected=p.detach().clone();expectedopt=optimizer_tensor_digest(opt.state_dict())
                replay=torch.load(file,weights_only=False)
                q=torch.nn.Parameter(replay['adapter'].clone());other=adam(q);other.load_state_dict(replay['optimizer'])
                restore_rng(torch,np,replay['rng']);step(q,other)
                self.assertTrue(torch.equal(expected,q));self.assertEqual(expectedopt,optimizer_tensor_digest(other.state_dict()))

    def test_sampling_uses_outer_start_identity_even_midround_resume(self):
        c=config();fresh=dict(outer_index=0,explicit_fresh=True,cumulative_optimizer_updates_before=0,previous_complete_checkpoint=None,
          sampling_model_identity=dict(source_model_identity='s'*64,checkpoint=None,optimizer_updates=0,adapter_tensor_sha256='ZERO_EFFECT_FRESH_LORA'))
        verify_sampling_identity(fresh,c)
        ref=dict(path='/old/manifest',sha256='p'*64)
        prior=dict(optimizer_updates=7,all_rank_adapter_sha256=['a'*64]*8,cursor=dict(outer_index=0,next_pair_cursor=33,usable_pair_count=33))
        nextouter=dict(outer_index=1,explicit_fresh=False,cumulative_optimizer_updates_before=7,previous_complete_checkpoint=ref,
          sampling_model_identity=dict(source_model_identity='s'*64,checkpoint=ref,optimizer_updates=7,adapter_tensor_sha256='a'*64))
        verify_sampling_identity(nextouter,c,prior)
        # Mid-round new adapter is intentionally NOT substituted as sampling identity.
        bad=copy.deepcopy(nextouter);bad['sampling_model_identity']['optimizer_updates']=8
        with self.assertRaises(ValueError):verify_sampling_identity(bad,c,prior)
        bad=copy.deepcopy(nextouter);bad['sampling_model_identity']['adapter_tensor_sha256']='b'*64
        with self.assertRaises(ValueError):verify_sampling_identity(bad,c,prior)

    def test_rank_state_semantic_metadata_refuses_stale_rng_or_counter(self):
        p=torch.nn.Parameter(torch.tensor([1.]));opt=adam(p);p.sum().backward();opt.step()
        state=state_to_cpu(opt.state_dict());digest=optimizer_tensor_digest(state);c=config()
        manifest=dict(run_id='new',optimizer_updates=1,cumulative_pair_exposure=32,cursor=dict(outer_index=0,next_pair_cursor=32,usable_pair_count=65),source_model_identity='s'*64,runtime_identity='r'*64,training_input=dict(pairs='pair-proof'),all_rank_adapter_sha256=['a'*64]*8,all_rank_optimizer_tensor_sha256=[digest]*8)
        saved=dict(schema='opsd-online-rank-state-v1',rank=0,run_id='new',config_sha256='c'*64,optimizer_updates=1,cumulative_pair_exposure=32,cursor=manifest['cursor'],source_model_identity='s'*64,runtime_identity='r'*64,training_input=manifest['training_input'],adapter_tensor_sha256='a'*64,optimizer_tensor_sha256=digest,optimizer=state,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),cuda_all=[torch.get_rng_state()]))
        self.assertTrue(verify_rank_state(saved,manifest,0,c))
        bad=copy.deepcopy(saved);bad['optimizer_updates']=0
        with self.assertRaises(ValueError):verify_rank_state(bad,manifest,0,c)
        bad=copy.deepcopy(saved);del bad['rng']['numpy']
        with self.assertRaises(ValueError):verify_rank_state(bad,manifest,0,c)


    def test_rng_payload_validates_actual_fields_without_advancing_rng(self):
        before=random.getstate()
        rng=dict(python=before,numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),cuda_all=[torch.get_rng_state()])
        validate_rng(rng);self.assertEqual(before,random.getstate())
        for broken in [dict(rng,cuda_all=[]),dict(rng,torch_cpu=torch.tensor([1.])),dict(rng,python=None)]:
            with self.assertRaises((ValueError,TypeError)):validate_rng(broken)

    def test_committed_receipts_use_exact_cursor_not_failed_attempts(self):
        manifest=dict(run_id='new',optimizer_updates=9,cursor=dict(outer_index=1,next_pair_cursor=33))
        rows={}
        for count,update,actual in [(1,8,32),(2,9,1)]:
            pairs=[dict(pair_index=i,valid_tokens=2,token_log=dict(path=str(i),sha256='x')) for i in range(actual)]
            rows[str(count)]=dict(schema='opsd-online-real-update-v1',accepted=True,run_id='new',outer_index=1,
                optimizer_updates=update,global_actual_pairs=actual,global_valid_tokens=actual*2,
                all_rank_actual_token_metrics=[dict(rank=r,pairs=pairs[r::8]) for r in range(8)])
        load=lambda ref:(rows[ref['path']],ref)
        refs=[dict(path='1'),dict(path='2')]
        self.assertEqual(verify_committed_receipts(refs,manifest,7,load,lambda ref:ref),66)
        with self.assertRaises(ValueError):verify_committed_receipts(refs+[dict(path='failed-attempt')],manifest,7,load,lambda ref:ref)
        rows['2']['optimizer_updates']=8
        with self.assertRaises(ValueError):verify_committed_receipts(refs,manifest,7,load,lambda ref:ref)


    def test_actual_source_files_fresh_content_and_postload_metadata(self):
        from checkpoint import fileproof,checked_file,checked_json,atomic_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'source';root.mkdir()
            files=[]
            for i in range(12):
                path=root/('model-'+str(i));path.write_bytes(bytes([i]));files.append(fileproof(path))
            manifest=Path(temp)/'manifest.json';ref=atomic_json(manifest,dict(files=files))
            c=dict(source_model_manifest=ref,paths=dict(model=str(root)))
            proof=verify_source100(c,checked_json,checked_file);self.assertTrue(verify_source_metadata(proof))
            (root/'model-0').write_bytes(b'z')
            with self.assertRaises(ValueError):verify_source100(c,checked_json,checked_file)
            with self.assertRaises(ValueError):verify_source_metadata(proof)


    def test_atomic_directory_manifest_failure_preserves_previous_and_new_attempt(self):
        from checkpoint import atomic_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'update-0';old.mkdir();(old/'manifest.json').write_text('old-complete')
            pending=root/'update-1.pending-first';(pending/'adapter').mkdir(parents=True)
            (pending/'rank-0-state.pt').write_bytes(b'actual-state')
            dest=root/'update-1'
            def fail(*args):raise OSError('full disk during pending manifest')
            with self.assertRaises(OSError):atomic_checkpoint_directory(pending,dest,dict(accepted=True),fail)
            self.assertFalse(dest.exists());self.assertTrue(pending.exists());self.assertEqual((old/'manifest.json').read_text(),'old-complete')
            retry=root/'update-1.pending-second';(retry/'adapter').mkdir(parents=True)
            atomic_checkpoint_directory(retry,dest,dict(accepted=True),atomic_json)
            self.assertTrue((dest/'manifest.json').exists());self.assertTrue(pending.exists())
            with self.assertRaises(ValueError):atomic_checkpoint_directory(pending,dest,{},atomic_json)


    def test_original_bare_runtime_manifest_and_wrapped_catalog(self):
        from checkpoint import fileproof,checked_file,atomic_json
        import sys
        from test_objective import require_actual_frozen_runtime
        runtime=require_actual_frozen_runtime()
        entries={name:dict(sha256=fileproof(runtime/name)['sha256']) for name in ('search_e1/core.py','search_e1/opsd.py')}
        with tempfile.TemporaryDirectory() as temp:
            for shape in [entries,dict(files=entries)]:
                manifest=Path(temp)/'manifest.json';ref=atomic_json(manifest,shape)
                oldpath=list(sys.path)
                try:
                    c=dict(runtime_manifest=ref,paths=dict(runtime=str(runtime)))
                    # This fixture isolates bare/wrapped catalog parsing; scorer itself has dedicated actual-QA fixtures.
                    import types
                    with patch.dict(sys.modules,{'scoring':types.SimpleNamespace(verify_scoring=lambda config:dict(binding='called-before-core-import'))}):
                        module=use_frozen_runtime(c,checked_file)
                    self.assertEqual(Path(module.__file__).resolve(),runtime/'search_e1/core.py')
                    self.assertEqual(c['_scoring_proof'],dict(binding='called-before-core-import'))
                    self.assertEqual(sys.path[:2],[str(runtime/'upstream/Search-R1'),str(runtime)])
                finally:sys.path[:]=oldpath


    def test_actual_frozen_core_and_nested_strict_scorer_integration(self):
        from checkpoint import fileproof,checked_file,atomic_json
        from scoring import QA_RELATIVE,INIT_RELATIVE,QA_SHA256
        import sys
        from test_objective import require_actual_frozen_runtime
        runtime=require_actual_frozen_runtime()
        entries={name:dict(sha256=fileproof(runtime/name)['sha256']) for name in ('search_e1/core.py','search_e1/opsd.py',QA_RELATIVE,INIT_RELATIVE)}
        oldpath=list(sys.path)
        saved={name:sys.modules.pop(name) for name in ['search_r1','search_r1.qa_protocol'] if name in sys.modules}
        try:
            with tempfile.TemporaryDirectory() as temp:
                ref=atomic_json(Path(temp)/'runtime.json',entries)
                c=dict(runtime_manifest=ref,paths=dict(runtime=str(runtime)))
                core=use_frozen_runtime(c,checked_file)
                proof=c['_scoring_proof'];self.assertTrue(proof['accepted'])
                qa=next(f for f in proof['actual_imported_scoring_files'] if f['path'].endswith('qa_protocol.py'))
                self.assertEqual(qa['sha256'],QA_SHA256)
                valid='<think>reason</think><answer>target</answer>'
                nested='<search><answer>target</answer></search>'
                self.assertEqual(core.reward(valid,['target'],[[0,len(valid)]]),1)
                self.assertEqual(core.reward(nested,['target'],[[0,len(nested)]]),0)
        finally:
            for name in ['search_r1','search_r1.qa_protocol']:sys.modules.pop(name,None)
            sys.modules.update(saved);sys.path[:]=oldpath


if __name__=='__main__':unittest.main()
