"""Local CPU fixtures. Synthetic200 is never signed or actual acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest

HERE=Path(__file__).resolve().parent
H=HERE.parent
REPO=H.parent/'opsd-unclipped-multitraj200-teacherrecompute'


def module(path,name):
    value=types.ModuleType(name);value.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec',dont_inherit=True,optimize=0),value.__dict__)
    return value


T=module(HERE/'consume_final200_logs_terminal.py','candidate_final200_logs')
C=module(HERE/'common.py','candidate_final200_common')
GATE_PATH=REPO/'search_r1/baselines/checkpoint_gate.py'
assert hashlib.sha256(GATE_PATH.read_bytes()).hexdigest()==C.GATE
G=module(GATE_PATH,'exact_fcec_gate')
RAW=H/'job3304-first5-complete-originals-v3/raw'
ACTUAL_DRIVER=(RAW/'driver-current-prefix.out').read_bytes()
first=next(json.loads(line.split(T.OUTER,1)[1]) for line in ACTUAL_DRIVER.decode().splitlines() if T.OUTER in line)
MANIFEST=json.loads((RAW/'full_state/global_step_5/manifest.json').read_bytes())
WORKERS=[json.loads((RAW/f'checkpoints/actor/global_step_5/training_state/rank_{i}.json').read_bytes())['pid'] for i in range(8)]
FSDP=next(RAW.glob('native-source/*fully_sharded_data_parallel.py')).read_bytes()
assert hashlib.sha256(FSDP).hexdigest()==G.FSDP_WARNING_SHA256
# The dummy receipt hash is strictly a direct-function fixture. main never runs,
# no envelope/key is created, and real accepted_context is not patched.
CONTEXT=dict(manifest=MANIFEST,workers=[dict(rank=i,pid=p) for i,p in enumerate(WORKERS)],
    accepted5=dict(manifest_path=str(C.RUN/'full_state/global_step_5/manifest.json'),manifest_sha256=C.ACCEPTED5,receipt_sha256='f'*64))


def emetrics(counts):
    result={};total=sum(counts.values())
    for source,n in counts.items():
        result['val/qa_count/'+source]=n;result['val/evaluated_count/'+source]=n
        result['val/test_score/'+source]=1/n
        for key in T.MEANS:result['val/'+key+'/'+source]=1/n if key in ('em','answer_format','format_rate') else (0.25 if key=='f1' else 2.0)
    result['val/qa_count/all']=total
    for key in T.MEANS:result['val/'+key+'/all']=sum(result['val/'+key+'/'+s]*n for s,n in counts.items())/total
    return result


def console(step,m):return 'step:'+str(step)+' - '+' - '.join(k+':'+format(v,'.3f') for k,v in m.items())
def event(marker,value):return marker+json.dumps(value,sort_keys=True,allow_nan=False)


def synthetic200():
    lines=[];initial=emetrics(T.DEV_COUNTS);lines.extend([event(T.EVAL,initial),console(0,initial)])
    for step in range(1,201):
        row=copy.deepcopy(first);row['actual_outer_updates']=row['global_step']=step;m=row['metrics']
        m['training/actual_outer_updates']=step;m['actor/lr']=1e-6*min(step/57,1)
        m={k:v for k,v in m.items() if not k.startswith(('val/','opsd_stage/'))};row['metrics']=m
        for rank in range(8):
            m[f'actor/optimizer_lr_used_mean/rank_{rank}']=1e-6*min((step-1)/57,1)
            m[f'teacher/cuda_update_entries_cumulative_per_rank/rank_{rank}']=float(step)
        if step%50==0:
            evaluation=emetrics(T.DEV_COUNTS);lines.append(event(T.EVAL,evaluation));m.update(evaluation)
        if step==5 or step%50==0:
            evaluation=emetrics(T.STAGE_COUNTS);lines.append(event(T.EVAL,evaluation));m.update({'opsd_stage/'+k:v for k,v in evaluation.items()})
        lines.extend([console(step,m),event(T.OUTER,row)])
        if step==5:
            lines.append(event(T.PAUSE,dict(driver_pid=MANIFEST['driver_pid'],job_id='3304',manifest_path=CONTEXT['accepted5']['manifest_path'],manifest_sha256=C.ACCEPTED5,pause_nonce=MANIFEST['pause_nonce'])))
            lines.append(event(T.RELEASE,dict(driver_pid=MANIFEST['driver_pid'],actual_outer_updates=5,next_global_step=6,receipt_sha256='f'*64)))
    final=emetrics(T.FINAL_COUNTS);lines.extend([event(T.EVAL,final),event(T.FINAL,final),console(200,final)])
    return ('\n'.join(lines)+'\n').encode()


class LogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.raw=synthetic200();cls.parsed=T.parse_log(cls.raw,CONTEXT,G,FSDP)
    def clone(self):return copy.deepcopy(self.parsed)
    def row(self,p,step):return next(r['value'] for r in p['records'] if r['marker']==T.OUTER and r['value']['global_step']==step)
    def test_synthetic_200_budget_and_11_evaluations(self):
        v=T.check_log(self.parsed,CONTEXT,G);self.assertEqual((v['actual_outer_updates'],v['adam_calls_per_rank'],len(v['evaluations'])),(200,2000,11));self.assertFalse(v['evaluations'][-1]['per_question_raw_evidence_available'])
    def test_actual_five_prefix_parser_replay_only(self):
        p=T.parse_log(ACTUAL_DRIVER[:ACTUAL_DRIVER.rfind(b'\n')+1],CONTEXT,G,FSDP)
        self.assertEqual(len([r for r in p['records'] if r['marker']==T.OUTER]),5)
        with self.assertRaisesRegex(ValueError,'Exactly actual1..200'):T.check_log(p,CONTEXT,G)
    def test_actual_aggregate_first5_owned_amp_frames(self):
        raw=(RAW/'train.log').read_bytes();p=T.parse_log(raw[:raw.rfind(b'\n')+1],CONTEXT,G,FSDP)
        self.assertEqual([r['value']['global_step'] for r in p['records'] if r['marker']==T.OUTER],[1,2,3,4,5])
    def test_missing_outer_rejected(self):
        p=self.clone();p['records']=[r for r in p['records'] if not(r['marker']==T.OUTER and r['value']['global_step']==50)]
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_duplicate_outer_rejected(self):
        p=self.clone();p['records'].append(copy.deepcopy(next(r for r in p['records'] if r['marker']==T.OUTER)))
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_executed201_rejected(self):
        p=self.clone();self.row(p,200)['actual_outer_updates']=201
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_bool_clock_rejected(self):
        p=self.clone();self.row(p,1)['global_step']=True
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_wrong_optimizer_calls_rejected(self):
        p=self.clone();self.row(p,9)['metrics']['actor/optimizer_steps/rank_7']=9
        with self.assertRaises(RuntimeError):T.check_log(p,CONTEXT,G)
    def test_wrong_sampling_count_rejected(self):
        p=self.clone();self.row(p,9)['metrics']['rollout/trajectories']=512
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_wrong_warmup_used_lr_rejected(self):
        p=self.clone();self.row(p,2)['metrics']['actor/optimizer_lr_used_mean/rank_2']=1e-6
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_pg_objective_rejected(self):
        p=self.clone();self.row(p,9)['metrics']['actor/pg_loss']=0
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_nonfinite_rank_loss_rejected(self):
        p=self.clone();self.row(p,9)['metrics']['actor/opsd_loss_mean/rank_7']=float('inf')
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_wrong_driver_and_config_rejected(self):
        for key,val in [('driver_pid',123),('config_sha256','0'*64)]:
            p=self.clone();self.row(p,9)[key]=val
            with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_unknown_suffix_duplicatekey_nonfinite_parser_rejected(self):
        for line in [event(T.OUTER,first)+' unknown',T.OUTER+'{"x":1,"x":2}',T.OUTER+'{"x":NaN}',event(T.OUTER,first)+event(T.OUTER,first)]:
            with self.assertRaises(ValueError):T.parse_log((line+'\n').encode(),CONTEXT,G,FSDP)
    def test_owned_exact_fsdp_warning_accepted(self):
        line=event(T.OUTER,first)+' (WorkerDict pid='+str(WORKERS[0])+') /mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: '+G.FSDP_WARNING_TEXT
        self.assertEqual(T.parse_log((line+'\n').encode(),CONTEXT,G,FSDP)['records'][0]['value'],first)
        with self.assertRaises(ValueError):T.parse_log((line.replace(str(WORKERS[0]),'999')+'\n').encode(),CONTEXT,G,FSDP)
    def test_console_rounding_change_rejected(self):
        p=self.clone();p['consoles'][0]['values']['val/em/all']='0.999'
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_signed_release_hash_mismatch_rejected(self):
        p=self.clone();next(r for r in p['records'] if r['marker']==T.RELEASE)['value']['receipt_sha256']='0'*64
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_wrong_eval_source_and_integer_hit_rejected(self):
        for m in [dict(emetrics(T.FINAL_COUNTS),**{'val/qa_count/bamboogle':124}),dict(emetrics(T.DEV_COUNTS),**{'val/em/nq':.01234})]:
            with self.assertRaises((ValueError,RuntimeError)):T.evaluation_math(m,T.FINAL_COUNTS if 'val/qa_count/bamboogle' in m else T.DEV_COUNTS,G)
    def test_wrong_eval_weighted_f1_rejected(self):
        m=emetrics(T.FINAL_COUNTS);m['val/f1/all']=.9
        with self.assertRaises(ValueError):T.evaluation_math(m,T.FINAL_COUNTS,G)
    def test_missing_final_source_rejected(self):
        m=emetrics(T.FINAL_COUNTS);del m['val/evaluated_count/musique']
        with self.assertRaises((ValueError,RuntimeError)):T.evaluation_math(m,T.FINAL_COUNTS,G)
    def test_eval_outer_full_value_pairing(self):
        p=self.clone();self.row(p,50)['metrics']['val/generated_tokens/all']=3
        with self.assertRaises(ValueError):T.check_log(p,CONTEXT,G)
    def test_full_logs_incomplete_tail_rejected(self):
        with self.assertRaises(ValueError):T.parse_log(self.raw[:-1],CONTEXT,G,FSDP)
    def test_changed_common_pin_rejected_before_compile(self):
        with self.assertRaises(ValueError):T.load_common('0'*64)


class TerminalTests(unittest.TestCase):
    def test_missing_all_is_blocked(self):
        v=T.terminal_layers({},C);self.assertIsNone(v['training_child_exit']['returncode']);self.assertIsNone(v['scheduler']['state']);self.assertTrue(v['resource_release']['status'].startswith('BLOCKED'))
    def test_child0_wrapper1_scheduler_failed_remain_separate(self):
        v=T.terminal_layers(dict(training_child_exit=b'{"returncode":0}',wrapper_exit=b'1\n',scheduler_original=b'JobId=3304 UserId=root(0) JobState=FAILED ExitCode=1:0\n'),C)
        self.assertEqual((v['training_child_exit']['returncode'],v['wrapper_exit']['returncode'],v['scheduler']['state']),(0,1,'FAILED'))
    def test_slurm_running_foreign_uid_and_agedout_rejected(self):
        for raw in [b'JobId=3304 UserId=root(0) JobState=RUNNING ExitCode=0:0',b'JobId=3304 UserId=user(501) JobState=FAILED ExitCode=1:0',b'Invalid job id specified']:
            with self.assertRaises(ValueError):T.scheduler(raw)
    def test_bool_exit_rejected(self):
        with self.assertRaises(ValueError):T.terminal_layers(dict(training_child_exit=b'{"returncode":false}'),C)
    def test_json_release_pass_not_resource_release(self):
        v=T.terminal_layers(dict(release_observation=b'{"status":"PASS","gpu_apps":[]}'),C)
        self.assertTrue(v['resource_release']['status'].startswith('BLOCKED'));self.assertIsNone(v['resource_release']['actual'])
    def test_actual_bad66_immutable_prefix_is_failed(self):
        p=H/'job3304-first5-resource-phase-originals-v3/all-resources.ndjson'
        with p.open('rb') as f:lines=[next(f) for _ in range(66)]
        uuids={v['uuid'] for v in json.loads(lines[0])['gpus']}
        r=T.resource_observation(lines,C,uuids)
        self.assertEqual(r['whole_rows'],66);self.assertTrue(r['status'].startswith('FAILED_WHOLE'));self.assertEqual(r['bad66']['line_sha256'],hashlib.sha256(lines[65]).hexdigest())
        v=json.loads(lines[65]);v['gpu_processes']=[a for a in v['gpu_processes'] if a.get('pid')!=2544824];lines[-1]=json.dumps(v).encode()+b'\n'
        with self.assertRaises(ValueError):T.resource_observation(lines,C,uuids)


if __name__=='__main__':unittest.main(verbosity=2)
