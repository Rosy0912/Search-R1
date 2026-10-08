"""Local CPU counterexamples only; no synthetic HF/training acceptance."""
import copy,importlib.util,json,math,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('step5',Path(__file__).with_name('step5-extract.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def fixtures():
    train=[];ranks=[]
    for step in range(1,6):
        metrics={'bapo/actual_outer_updates':step,'actor/pg_loss':.1,'actor/entropy_loss':.2,'actor/grad_norm':.3,
            'bapo/stage_validation':step/10,'timing_s/step':100,'bapo/loss_tokens':100,'bapo/grad_norm_min':.2,
            'bapo/grad_norm_max':.3,'actor/optimizer_steps':20,'bapo/stage_cost/trajectories':64,
            'bapo/stage_evaluated_count':64,'bapo/early_stage':1,'bapo/stalled':0,**{key:0 for key in m.AUDITS}}
        train.append({'step':step,'metrics':metrics})
        for rank in range(8):ranks.append(dict(rank=rank,scheduler_last_epoch=step,optimizer_steps=20,
            lr_before=1e-6*(step-1)/57,lr_after=1e-6*step/57,metrics={'actor/pg_loss':[.1]*20,'actor/entropy_loss':[.2]*20,'actor/grad_norm':[.3]*20}))
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
        import runpy,sys
        from unittest.mock import patch
        repo=Path(__file__).resolve().parents[1]/'bapo-e1-300-countfix';sys.path.insert(0,str(repo))
        actual=runpy.run_path(str(repo/'tests/baselines/test_sequence_mean_actor.py'))
        with patch.object(actual['TensorDictBase'],'cuda',lambda self,*args,**kw:self):
            gradients,raw=actual['actor'](actual['fixture'](320),16,2,entropy=0.)
        self.assertEqual(len(gradients),20);self.assertEqual(raw['actor/optimizer_steps'],20)
        for key in ('actor/pg_loss','actor/entropy_loss','actor/grad_norm'):
            self.assertEqual(len(raw[key]),20);self.assertTrue(all(math.isfinite(x)for x in raw[key]))
        train,ranks=fixtures()
        for event in ranks:event['metrics']=copy.deepcopy(raw)
        m.validate_metrics(train,ranks)
        for kind in ('scalar','nineteen','twentyone','nan'):
            bad=copy.deepcopy(ranks)
            if kind=='scalar':bad[0]['metrics']['actor/pg_loss']=.1
            if kind=='nineteen':bad[0]['metrics']['actor/pg_loss'].pop()
            if kind=='twentyone':bad[0]['metrics']['actor/pg_loss'].append(.1)
            if kind=='nan':bad[0]['metrics']['actor/pg_loss'][-1]=float('nan')
            with self.assertRaises(ValueError):m.validate_metrics(train,bad)

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
