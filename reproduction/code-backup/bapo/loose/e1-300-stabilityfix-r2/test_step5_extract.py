"""Local CPU counterexamples only; no synthetic HF/training acceptance."""
import copy,importlib.util,json,math,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('step5',Path(__file__).with_name('step5-extract.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def sync_driver(train,ranks):
    for event in train:
        step=event['step'];raw=[r['metrics']for r in ranks if r['scheduler_last_epoch']==step];driver=event['metrics']
        for key in ('actor/pg_loss','actor/kl_loss','actor/entropy_loss','actor/ppo_kl','actor/pg_clipfrac'):
            driver[key]=math.fsum(x for r in raw for x in r[key])/160
        driver['actor/grad_norm']=math.fsum(raw[0]['actor/grad_norm'])/20
        driver['mfu/actor']=math.fsum(r['mfu/actor']for r in raw)/8
        for key in ('bapo/loss_tokens','bapo/old_logprob_nonfinite','bapo/new_logprob_nonfinite'):
            driver[key]=sum(r[key]for r in raw)
        for key in ('actor/optimizer_steps','actor/skipped_empty_minibatches','actor/lr','actor/lr_before_outer',
                    'actor/scheduler_outer_steps','actor/kl_coef','bapo/grad_norm_min','bapo/grad_norm_max'):
            driver[key]=raw[0][key]


def fixtures():
    train=[];ranks=[]
    for step in range(1,6):
        driver={'bapo/actual_outer_updates':step,'bapo/stage_validation':step/10,'timing_s/step':100,
            'bapo/stage_cost/trajectories':64,'bapo/stage_evaluated_count':64,
            'bapo/early_stage':1,'bapo/stalled':0,**{key:0 for key in m.AUDITS}}
        train.append({'step':step,'metrics':driver})
        for rank in range(8):
            raw={'actor/pg_loss':[.1+rank]*20,'actor/kl_loss':[.2+rank]*20,'actor/entropy_loss':[.3+rank]*20,
                'actor/ppo_kl':[.01+rank]*20,'actor/pg_clipfrac':[.1]*20,
                'actor/grad_norm':[.2]*10+[.3]*10,'bapo/grad_norm_min':.2,'bapo/grad_norm_max':.3,
                'actor/optimizer_steps':20,'actor/skipped_empty_minibatches':0,'actor/lr':1e-6*step/114,
                'actor/lr_before_outer':1e-6*(step-1)/114,'actor/scheduler_outer_steps':step,'actor/kl_coef':.001,
                'mfu/actor':.2+rank/100,'bapo/loss_tokens':100+rank,'bapo/old_logprob_nonfinite':0,'bapo/new_logprob_nonfinite':0}
            ranks.append(dict(rank=rank,scheduler_last_epoch=step,optimizer_steps=20,
                lr_before=1e-6*(step-1)/114,lr_after=1e-6*step/114,metrics=raw))
    sync_driver(train,ranks)
    return train,ranks

class Contract(unittest.TestCase):
    def test_actual_metrics_happy(self):
        t,r=fixtures();self.assertEqual(m.validate_metrics(t,r),dict(best=.5,stalled=0,early=True,patience=5))
    def test_all_five_boundary_audit_clock_counterexamples(self):
        for kind in ['four','six','duplicate','audit','nan','stage','adam','rankduplicate','rankmissing','rankadam','clock','lateclock','replay','outerbool']:
            with self.subTest(kind=kind):
                t,r=fixtures()
                if kind=='four':t.pop()
                if kind=='six':t.append(copy.deepcopy(t[-1]));t[-1]['step']=6
                if kind=='duplicate':t[-1]=copy.deepcopy(t[0])
                if kind=='audit':t[3]['metrics'][m.AUDITS[-1]]=1
                if kind=='nan':t[3]['metrics']['actor/pg_loss']=float('nan')
                if kind=='stage':t[3]['metrics']['bapo/stage_evaluated_count']=63
                if kind=='adam':t[3]['metrics']['actor/optimizer_steps']=19
                if kind=='rankduplicate':r[-1]=copy.deepcopy(r[0])
                if kind=='rankmissing':r.pop()
                if kind=='rankadam':r[0]['optimizer_steps']=19
                if kind=='clock':r[0]['lr_after']=1e-6/85
                if kind=='lateclock':r[-1]['scheduler_last_epoch']=6
                if kind=='replay':t[4]['metrics']['bapo/stalled']=4
                if kind=='outerbool':t[0]['metrics']['bapo/actual_outer_updates']=True
                with self.assertRaises(ValueError):m.validate_metrics(t,r)
    def test_real_actor_produces_rank_mini_lists_and_rejects_bad_shapes(self):
        import runpy,sys,os
        from unittest.mock import patch
        repo=Path(os.environ.get('STEP5_EXTRACT_TEST_REPO',str(Path(__file__).resolve().parents[1]/'bapo-e1-300-stabilityfix')));sys.path.insert(0,str(repo))
        actual=runpy.run_path(str(repo/'tests/baselines/test_sequence_mean_actor.py'))
        with patch.object(actual['TensorDictBase'],'cuda',lambda self,*args,**kw:self),patch.object(actual['torch'].cuda,'current_device',return_value=0):
            gradients,raw=actual['actor'](actual['fixture'](320),16,2,entropy=0.)
        self.assertEqual(len(gradients),20);self.assertEqual(raw['actor/optimizer_steps'],20)
        for key in ('actor/pg_loss','actor/entropy_loss','actor/grad_norm'):
            self.assertEqual(len(raw[key]),20);self.assertTrue(all(math.isfinite(x)for x in raw[key]))
        train,ranks=fixtures()
        for event in ranks:
            fresh=copy.deepcopy(raw);fresh['mfu/actor']=.1
            for key in ('actor/lr','actor/lr_before_outer','actor/scheduler_outer_steps'):
                fresh[key]=event['metrics'][key]
            event['metrics']=fresh
        sync_driver(train,ranks)
        m.validate_metrics(train,ranks)
        for kind in ('scalar','nineteen','twentyone','nan'):
            bad=copy.deepcopy(ranks)
            if kind=='scalar':bad[0]['metrics']['actor/pg_loss']=.1
            if kind=='nineteen':bad[0]['metrics']['actor/pg_loss'].pop()
            if kind=='twentyone':bad[0]['metrics']['actor/pg_loss'].append(.1)
            if kind=='nan':bad[0]['metrics']['actor/pg_loss'][-1]=float('nan')
            with self.assertRaises(ValueError):m.validate_metrics(train,bad)

    def test_allrank_spike_and_false_rank0_summary_rejected(self):
        train,ranks=fixtures()
        ranks[7]['metrics']['actor/pg_loss'][-1]=1e9
        sync_driver(train,ranks)
        proof=m.validate_allrank_reporting(train,ranks)
        self.assertGreater(proof[0]['expected_driver']['actor/pg_loss'],1e6)
        train[0]['metrics']['actor/pg_loss']=math.fsum(ranks[0]['metrics']['actor/pg_loss'])/20
        with self.assertRaisesRegex(ValueError,'allrank/globalmini'):
            m.validate_metrics(train,ranks)

    def test_sum_counts_replica_and_mini_raw_counterexamples(self):
        for kind in ('countmean','stepssum','driverlr','rawlr','rawnorm','klrank0','old57','missingkl','rawskipped'):
            with self.subTest(kind=kind):
                train,ranks=fixtures();d=train[0]['metrics'];r=ranks[0]['metrics']
                if kind=='countmean':d['bapo/loss_tokens']/=8
                if kind=='stepssum':d['actor/optimizer_steps']*=8
                if kind=='driverlr':d['actor/lr']=1e-6/57
                if kind=='rawlr':r['actor/lr']=1e-6/57
                if kind=='rawnorm':r['actor/grad_norm'][0]=99
                if kind=='klrank0':d['actor/kl_loss']=.2
                if kind=='old57':ranks[0]['lr_after']=1e-6/57
                if kind=='missingkl':r.pop('actor/kl_loss')
                if kind=='rawskipped':r['actor/skipped_empty_minibatches']=1
                with self.assertRaises(ValueError):m.validate_metrics(train,ranks)

    def test_exact_source_warmup_root_and_objective_config(self):
        raw=Path(__file__).with_name('config-fixture-from-actual3177.json')
        c=json.loads(raw.read_text());run=m.ROOT/'runs/bapo-instruct300-formal300-9999'
        old='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004'
        def rewrite(value):
            if isinstance(value,str):return value.replace(old,str(m.ROOT))
            if isinstance(value,dict):return {k:rewrite(v)for k,v in value.items()}
            if isinstance(value,list):return [rewrite(v)for v in value]
            return value
        c=rewrite(c);c['baselines']['algorithm_source_commit']=m.EXPECTED_SOURCE
        c['actor_rollout_ref']['actor']['optim']['num_warmup_steps']=114
        c['trainer']['default_local_dir']=str(run/'checkpoints')
        c['trainer']['e1_gate_receipt']=str(run/'logs/step5-independent-release.json')
        m.validate_config(c,run)
        for kind in ('source','old57','warmupratio','style','model','tp','sp','epochs','minibatch'):
            with self.subTest(kind=kind):
                bad=copy.deepcopy(c);a=bad['actor_rollout_ref']['actor']
                if kind=='source':bad['baselines']['algorithm_source_commit']='733097feeae371b4efd8fd48940c46ff36da839b'
                if kind=='old57':a['optim']['num_warmup_steps']=57
                if kind=='warmupratio':a['optim']['lr_warmup_steps_ratio']=.285
                if kind=='style':a['optim']['warmup_style']='cosine'
                if kind=='model':bad['actor_rollout_ref']['model']['path']=old+'/assets/qwen2.5-3b-instruct-aa8e725'
                if kind=='tp':bad['actor_rollout_ref']['rollout']['tensor_model_parallel_size']=2
                if kind=='sp':a['ulysses_sequence_parallel_size']=2
                if kind=='epochs':a['ppo_epochs']=2
                if kind=='minibatch':a['ppo_mini_batch_size']=64
                with self.assertRaises(ValueError):m.validate_config(bad,run)

    def test_new_checkpoint_identity_rejects_old_receipt_bindings(self):
        run=m.ROOT/'runs/bapo-instruct300-formal300-9999'
        manifest=dict(run_root=str(run),source_commit=m.EXPECTED_SOURCE,global_step=5,
                      actual_outer_updates=5,horizon=300,warmup_steps=114)
        m.validate_checkpoint_identity(manifest,run)
        for key,value in [('warmup_steps',57),('source_commit','733097feeae371b4efd8fd48940c46ff36da839b'),
                          ('run_root','/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177'),
                          ('actual_outer_updates',2),('horizon',500)]:
            bad=dict(manifest,**{key:value})
            with self.assertRaises(ValueError):m.validate_checkpoint_identity(bad,run)

    def test_native_hf_tokenizer_and_resource_guards_not_weakened(self):
        import ast,hashlib
        ref=json.loads(Path(__file__).with_name('reference-guards.json').read_text())
        tree=ast.parse(Path(m.__file__).read_text())
        actual={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
                for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ref['guards']}
        self.assertEqual(actual,ref['guards'])

    def test_prefix_snapshot_exact_immutable_no_overwrite(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);live=root/'train.log';out=root/'prefix.log'
            live.write_text('before\nE1_STEP5_PAUSED_JSON={"pid":3}\nappend warning\n')
            proof=m.capture_prefix(live,out,'E1_STEP5_PAUSED_JSON=');self.assertEqual(out.read_text(),'before\nE1_STEP5_PAUSED_JSON={"pid":3}\n')
            self.assertEqual(proof['sha256'],m.digest(out))
            with self.assertRaises(FileExistsError):m.capture_prefix(live,out,'E1_STEP5_PAUSED_JSON=')
    def test_prefix_missing_duplicate_partial_rejected(self):
        for text in ['missing\n','E1_STEP5_PAUSED_JSON={}\nE1_STEP5_PAUSED_JSON={}\n','E1_STEP5_PAUSED_JSON={}']:
            with tempfile.TemporaryDirectory()as d:
                p=Path(d);(p/'live').write_text(text)
                with self.assertRaises(ValueError):m.capture_prefix(p/'live',p/'out','E1_STEP5_PAUSED_JSON=')
    def test_scheduler_current_running_cost(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'slurm';p.write_text('JobId=39 JobState=RUNNING NodeList=h100-5 RunTime=1-02:03:04')
            self.assertEqual(m.scheduler_proof(p,39)['elapsed_seconds'],93784)
            for text in ['JobId=390 JobState=RUNNING NodeList=h100-5 RunTime=00:01:00','JobId=39 JobState=FAILED NodeList=h100-5 RunTime=00:01:00','JobId=39 JobState=RUNNING NodeList=a800 RunTime=00:01:00']:
                p.write_text(text)
                with self.assertRaises(ValueError):m.scheduler_proof(p,39)
    def test_resource_bounds_schema_whole_busy_is_compatible(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory()as d:
            root=Path(d).resolve();run=root/'run';run.mkdir()
            oldassets=root/'bapo/assets-root/assets';oldassets.mkdir(parents=True)
            newroot=root/'new';(newroot/'assets-root').mkdir(parents=True)
            (newroot/'assets-root/assets').symlink_to(oldassets,target_is_directory=True)
            local=dict(proxy_env=[],host='h100-5',paths=[dict(path=str(oldassets/name),mount='/dev/a ext4 /dataset')for name in ('wiki/e5_Flat.index','wiki/wiki-18.jsonl','e5-base-v2')])
            devices={str(i):dict(uuid='GPU-'+str(i),total_mib=80000)for i in range(8)}
            totals={str(i):dict(external_sm_percent=None,external_sm_upper_bound_percent=0,whole_gpu_sm_percent=100,external_memory_percent=0,sm_evidence='whole_gpu_upper_bound')for i in range(8)}
            record=dict(run=str(run),slurm_job_id='39',pause=False,time=1,allocated_gpus=devices,external_totals=totals)
            locality=root/'retrieval-assets-locality.json';locality.write_text(json.dumps(local))
            (root/'runtime-profile.txt').write_text('CUDA E5 GPU FAISS loopback127.0.0.1:18083')
            path=root/'resource-watch.jsonl';path.write_text(json.dumps(record)+'\n')
            with patch.object(m,'ROOT',newroot):
                self.assertEqual(len(m.resource_proof(root,run,39)['allocated_uuids']),8)
                for kind in ['missing','external','pause','wrongjob','duplicateuuid']:
                    bad=copy.deepcopy(record)
                    if kind=='missing':bad['external_totals'].pop('0')
                    if kind=='external':bad['external_totals']['0']['external_memory_percent']=16
                    if kind=='pause':bad['pause']=True
                    if kind=='wrongjob':bad['slurm_job_id']='40'
                    if kind=='duplicateuuid':bad['allocated_gpus']['0']['uuid']='GPU-1'
                    path.write_text(json.dumps(bad)+'\n')
                    with self.assertRaises(ValueError):m.resource_proof(root,run,39)
                path.write_text(json.dumps(record)+'\n')
                for kind in ['empty','duplicate','wrongasset','h100-50','filesystem']:
                    bad=copy.deepcopy(local)
                    if kind=='empty':bad['paths']=[]
                    if kind=='duplicate':bad['paths'][1]=copy.deepcopy(bad['paths'][0])
                    if kind=='wrongasset':bad['paths'][0]['path']='/dataset/another.index'
                    if kind=='h100-50':bad['host']='h100-50'
                    if kind=='filesystem':bad['paths'][0]['mount']='node:/remote nfs /dataset'
                    locality.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):m.resource_proof(root,run,39)
                local['host']='h100-5.example.test';locality.write_text(json.dumps(local));m.resource_proof(root,run,39)

    def test_real_tokenizer_pair_chat_pad_added_metadata_guards(self):
        from tokenizers import Tokenizer,AddedToken
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        def make():
            engine=Tokenizer(WordLevel({'[UNK]':0,'[EOS]':1,'[PAD]':2,'What':3,'is':4,'the':5,'capital':6,'of':7,'France':8,'?':9},unk_token='[UNK]'));engine.pre_tokenizer=Whitespace()
            tok=PreTrainedTokenizerFast(tokenizer_object=engine,unk_token='[UNK]',eos_token='[EOS]',pad_token='[PAD]')
            tok.chat_template="{{ messages[0]['content'] }}";return tok
        proof=m.tokenizer_proof(make(),make());self.assertTrue(proof['real_chat_template_tokenization_equal'])
        for kind in ('chat','pad','added'):
            left,right=make(),make()
            if kind=='chat':right.chat_template="{{ messages[0]['content'] }} changed"
            if kind=='pad':right.pad_token='[EOS]'
            if kind=='added':right.add_tokens([AddedToken('France',single_word=True,normalized=False)])
            self.assertEqual(left.get_vocab(),right.get_vocab())
            with self.assertRaises(ValueError):m.tokenizer_proof(left,right)

    def test_real_hf_missing_local_model_fails(self):
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaises((OSError,ValueError)):m.hf_proof(Path(d),Path(d))
    def test_no_pass_receipt_generated(self):
        text=Path(m.__file__).read_text()
        self.assertNotIn('PASS_INDEPENDENT_E1_STEP5_REVIEW',text)
        self.assertIn("status='REQUIRES_INDEPENDENT_REVIEW'",text)
        self.assertIn("result['cpu_hf']=hf_proof(MODEL,ck)",text)

if __name__=='__main__':unittest.main()
