"""Synthetic boundary counterexamples only; never actual training/HF acceptance."""
import copy, importlib.util, subprocess, sys, unittest, tempfile, hashlib
from pathlib import Path

def outer(i):
    m={k:0 for k in f.AUDITS}
    m.update({'bapo/actual_outer_updates':i,'actor/pg_loss':.01,'actor/entropy_loss':0.,'actor/grad_norm':1.,
              'bapo/stage_validation':.2,'state_tokens/total':100,'timing_s/step':80.,'actor/optimizer_steps':10,
              'bapo/loss_tokens':40,'bapo/grad_norm_min':.1,'bapo/grad_norm_max':2.,
              'bapo/stage_cost/trajectories':64,'bapo/stage_evaluated_count':64})
    return m

def fixture(n=None):
    n=f.EXPECTED_OUTERS if n is None else n
    events=[dict(marker='BAPO_TRAIN_METRICS_JSON',value=dict(step=i,metrics=outer(i))) for i in range(1,n+1)]
    m={'val/qa_count/all':51713,'val/cost/trajectories':51713,'val/em/all':.2,'val/f1/all':.3}
    for prefix in ('val/qa_count/','val/evaluated_count/'):
        m.update({prefix+k:v for k,v in f.EXPECTED_DATASETS.items()})
    if f.EXPECTED_OUTERS==500: events.append(dict(marker='SEARCH_R1_FINAL_METRICS_JSON',value=m))
    checks={str(i):{'status':'NUMERIC_SUMMARY_PRESENT'} for i in range(1,f.EXPECTED_OUTERS+1)}
    return events,checks
def checkpoint_fixture():
    # Synthetic metadata for validator counterexamples, never a real HF acceptance.
    names=['added_tokens.json','config.json','generation_config.json','merges.txt','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json']
    return dict(files=[dict(path=n,bytes=1,sha256='a'*64) for n in names],
        tokenizer_compatibility=dict(status='TOKENIZER_MATCH',full_vocabulary_equal=True,bos_equal=True,eos_equal=True,
            base_vocab_sha256='a'*64,checkpoint_vocab_sha256='a'*64,base_vocab_size=151665,checkpoint_vocab_size=151665,
            base_bos=None,checkpoint_bos=None,base_eos=151643,checkpoint_eos=151643),
        hf_load=dict(status='HF_LOAD_AND_FORWARD_OK',device='cpu',parameters=3085938688,nonfinite_parameters=[],
            tied_input_output_storage=True,loading_info={k:[] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')},
            forward=dict(logits_finite=True,device='cpu',input_tokens=5,logits_shape=[1,5,151936],next_token_id=1)),
        parameter_delta=dict(changed=True,max_abs_delta=.01,comparison='base_cast_to_checkpoint_dtype_then_float64',
            tied_weight_resolution=dict(status='QWEN2_TIED_DUPLICATE_EXACTLY_VERIFIED',base_logical_key='lm_head.weight',
                base_storage_key='model.embed_tokens.weight',checkpoint_extra_key='lm_head.weight',
                checkpoint_equal_key='model.embed_tokens.weight',shape=[151936,2048],elements_compared=311164928,exact_equal=True)))

class RewardfixToolTest(unittest.TestCase):
    def setUp(self):
        global f
        f=self.module
    def test_complete_count(self):
        e,c=fixture();self.assertEqual(f.validate_formal_events(e,c),[])
    def test_missing_and_extra(self):
        for n in (f.EXPECTED_OUTERS-1,f.EXPECTED_OUTERS+1):
            e,c=fixture(n);self.assertTrue(f.validate_formal_events(e,c))
    def test_duplicate_and_missing(self):
        e,c=fixture();e.insert(0,copy.deepcopy(e[0]));self.assertTrue(f.validate_formal_events(e,c))
        e,c=fixture();del e[0];self.assertTrue(f.validate_formal_events(e,c))
    def test_noninteger_step(self):
        e,c=fixture();e[0]['value']['step']=1.0;self.assertTrue(f.validate_formal_events(e,c))
    def test_audit_stage_grad_loss(self):
        for key,value in [(k,1) for k in f.AUDITS]+[('bapo/stage_evaluated_count',63),('bapo/loss_tokens',0),('bapo/grad_norm_min',0),('bapo/grad_norm_max',float('nan')),('actor/optimizer_steps',0)]:
            e,c=fixture();e[0]['value']['metrics'][key]=value;self.assertTrue(f.validate_formal_events(e,c),key)
    def test_state_source_and_count(self):
        n=f.EXPECTED_OUTERS
        s=dict(file='/run/algorithm/global_step_'+str(n)+'/state.json',value=dict(method='bapo',global_step=n,actual_outer_updates=n,algorithm_source_commit=f.EXPECTED_SOURCE,resume_supported=False,bapo=dict(patience=5,early=n<500,best=.3,stalled=0)))
        self.assertEqual(f.validate_expected_state([s],f.EXPECTED_SOURCE),[])
        for key,value in [('actual_outer_updates',n-1),('algorithm_source_commit','b20'),('global_step',n+1)]:
            bad=copy.deepcopy(s);bad['value'][key]=value;self.assertTrue(f.validate_expected_state([bad],f.EXPECTED_SOURCE))
    def test_hf_tokenizer_alias_contradictions(self):
        self.assertEqual(f.validate_checkpoint(checkpoint_fixture()),[])
        for category,key,value in [('hf_load','parameters',1),('hf_load','device','cuda'),('tokenizer_compatibility','full_vocabulary_equal',False),('tokenizer_compatibility','checkpoint_vocab_sha256','b'*64)]:
            ck=checkpoint_fixture();ck[category][key]=value;self.assertTrue(f.validate_checkpoint(ck))
        for key,value in [('logits_finite',False),('device','cuda'),('logits_shape',[1,5,99])]:
            ck=checkpoint_fixture();ck['hf_load']['forward'][key]=value;self.assertTrue(f.validate_checkpoint(ck))
        for key,value in [('status','fake'),('base_storage_key','other'),('shape',[151936,2047]),('elements_compared',1),('exact_equal',False)]:
            ck=checkpoint_fixture();ck['parameter_delta']['tied_weight_resolution'][key]=value;self.assertTrue(f.validate_checkpoint(ck))
    def test_missing_or_old_run_source_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):f.read_run_source_record(root)
            path=root/'source-commit.txt';path.write_text('b20f371972bb1bc41b735c6cd350b9a779d8820d\n')
            with self.assertRaises(ValueError):f.read_run_source_record(root)
            path.write_text(f.EXPECTED_SOURCE+'\n')
            record=f.read_run_source_record(root)
            self.assertEqual(record['source_commit'],f.EXPECTED_SOURCE)
            self.assertEqual(record['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
    def test_source_marker_guard_precedes_hf(self):
        source=self.tool.read_text()
        main=source[source.index('def main():'):]
        self.assertLess(main.index('run_source_record=read_run_source_record(root)'),main.index('AutoModelForCausalLM.from_pretrained'))
    def test_impossible_stalled_state_and_replay(self):
        n=f.EXPECTED_OUTERS
        state=dict(file='/run/algorithm/global_step_'+str(n)+'/state.json',value=dict(method='bapo',global_step=n,actual_outer_updates=n,algorithm_source_commit=f.EXPECTED_SOURCE,resume_supported=False,bapo=dict(patience=5,early=n<500,best=.2,stalled=n-1)))
        events,_=fixture();train=[e for e in events if e['marker']=='BAPO_TRAIN_METRICS_JSON']
        self.assertEqual(f.validate_expected_state([state],f.EXPECTED_SOURCE),[])
        self.assertEqual(f.validate_controller_replay([state],train),[])
        bad=copy.deepcopy(state);bad['value']['bapo']['stalled']=420 if n<500 else 500
        self.assertTrue(f.validate_expected_state([bad],f.EXPECTED_SOURCE))
        bad=copy.deepcopy(state);bad['value']['bapo']['best']=.3
        self.assertTrue(f.validate_controller_replay([bad],train))
    def test_job_binding(self):
        text='JobId=4000 JobState=COMPLETED ExitCode=0:0 RunTime=01:00:00'
        self.assertEqual(f.parse_scheduler_cost(text,4000)['issues'],[])
        self.assertTrue(f.parse_scheduler_cost(text,4001)['issues'])
    def test_fixed_root_rejects_old(self):
        proc=subprocess.run([sys.executable,str(self.tool),'--run-root','/dataset/shaoshitong/search-r1-rl/bapo/runs/old','--output','/tmp/no-output-bapo.json','--base-model','/tmp/nonexistent','--slurm-record','/tmp/nonexistent','--job-id','4000','--hf-load','--compare-weights','--hash-checkpoint'],capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0);self.assertIn('fixed corrected ROOT/runs',proc.stderr)
    def test_final_51713_only_formal(self):
        if f.EXPECTED_OUTERS!=500:
            e,c=fixture();self.assertFalse(any(x['marker']=='SEARCH_R1_FINAL_METRICS_JSON' for x in e));self.assertEqual(f.validate_formal_events(e,c),[]);return
        for change in ('missing','wrong','duplicate'):
            e,c=fixture()
            if change=='missing':e.pop()
            elif change=='wrong':e[-1]['value']['val/qa_count/all']=64
            else:e.append(copy.deepcopy(e[-1]))
            self.assertTrue(f.validate_formal_events(e,c))
for label in ('smoke2','accept5','formal500'):
    tool=Path(__file__).with_name('bapo-'+label+'-evidence-gpu64-tied.py')
    spec=importlib.util.spec_from_file_location(label,tool);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    globals()['Test_'+label]=type('Test_'+label,(RewardfixToolTest,),dict(module=module,tool=tool))
del RewardfixToolTest
if __name__=='__main__':unittest.main()
