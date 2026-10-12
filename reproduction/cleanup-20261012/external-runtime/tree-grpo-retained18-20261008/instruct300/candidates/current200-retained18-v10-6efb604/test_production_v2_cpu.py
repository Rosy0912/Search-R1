"""V2 real contract calls with tiny IO and injected scheduler, never real Slurm."""
import ast
import base64
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
import production_contract as c
import slurm_identity as s
import submit_once as submitter
import guard_entry
from test_runtime_contract_cpu import observation,write,identity,direct,allskip,checkpoint,HERE,package

ALLOCATION='JobId=99 JobState=RUNNING NodeList=h100-10 BatchHost=h100-10 NumNodes=1 OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=96 Requeue=0 Restarts=0 TimeLimit=3-00:00:00 MinMemoryNode=256G'

def cluster_proof():return dict(cluster_name=s.CLUSTER,local_queries=True,configuration_stdout='ClusterName = wuhan-h100\n',federation_stdout='')
def journal(path):return dict(status='SUBMITTED',cluster_name=s.CLUSTER,node_list=s.NODE,job_id=99,run_id='tree-fresh200-train-99',
    manifest_sha256='m',review_sha256='r',dispatch_intent_path=str(path),parsable_stdout='99;wuhan-h100',stdout='99;wuhan-h100\n',stderr='',returncode=0,local_cluster_identity=cluster_proof())
def scheduler(command,**kwargs):
    if command[-1]=='config':out='ClusterName = wuhan-h100\n'
    elif command[-1]=='federation':out=''
    else:out=ALLOCATION
    return N(returncode=0,stdout=out,stderr='')

def five_fixture(root):
    p,_=observation(root/'observation');run=root/'tree-fresh200-train-99';gate=run/'output/acceptance-step5'
    for outer in range(1,6):direct(run,outer,writes=3)
    ranks=c.direct_rank_records(run,p,'config',5);bindings={}
    config={'target_old_policy_kl':.2,'negative_tail_limit':.01,'sampling_probability_correction':{'initial_actor_repeat_max_abs_delta':0.}}
    for outer in range(1,6):
        for rank in range(8):
            path=run/'output/tree-rank-audit/policy-safety'/f'step-{outer:06d}-attempt-{outer:06d}-rank-{rank:02d}-event-0001.json'
            record=dict(outer_step=outer,attempt_id=outer,rank=rank,monitor_rows=32,config=config,
                probability_denominator='actual_behavior_mu',original_sampler_actor_consistency_accepted=False,
                decision=dict(allowed=True,sampled_old_policy_kl=.001,negative_tail_fraction=0.,
                              initial_actor_repeat_checked=True,global_monitor_actor_repeat_max_abs_delta=0.,reasons=[]))
            bindings[str(path)]=write(path,record)['sha256']
    proofs=dict(full_state_manifest_sha256='state',hf_reload_report_sha256='hfcheck',state_cpu_report_sha256='statecheck',behavior_mu_cpu_report_sha256='mucheck',
        partial_raw_cpu_report_sha256='rawcheck',development={'sha256':'dev'},rank_counts=ranks,
        tree_and_safety=dict(evidence_sha256=bindings,observed_refusal_events=0))
    ready=write(gate/'ready.json',dict(completed_outer_updates=5,next_outer_update=6,actual_update_limit=200,full_state_manifest_sha256='state',hf_identity={'sha256':'hf'}))
    assessment=dict(identity(run),status='INDEPENDENT_SAME_INSTANCE_FIRST5_THRESHOLDS_ACCEPTED',fork_turns='none',training_protocol=c.PROFILE,
        completed_outer=5,continue_range=[6,200],unchanged_thresholds_accepted=True,original_sampler_actor_gate_accepted=False,ready_sha256=ready['sha256'],
        bootstrap_thresholds_sha256=p['production_inputs']['threshold_acceptance']['sha256'],calibration_ready_sha256=p['production_inputs']['actual_calibration_ready']['sha256'],
        calibration_acceptance_sha256=p['production_inputs']['actual_calibration_acceptance']['sha256'],values=p['bootstrap_thresholds'],
        observations=c.short_threshold_observations(run,proofs,p['bootstrap_thresholds']),partial_raw_cpu_report_sha256='rawcheck',assessment_rationale='Synthetic CPU fixture, not actual training evidence')
    descriptor=write(gate/'external-threshold-assessment.json',assessment)
    review=dict(identity(run),status='PASS',fork_turns='none',ready_sha256=ready['sha256'],short_training_thresholds_accepted=True,threshold_assessment_sha256=descriptor['sha256'],
        partial_raw_cpu_report_sha256='rawcheck',full_state_reload_accepted=True,full_hf_reload_accepted=True,rank_tree_numerics_accepted=True,fixed512_evaluation_accepted=True,
        behavior_mu_binding_accepted=True,full_state_manifest_sha256='state',hf_identity_sha256='hf',hf_reload_report_sha256='hfcheck',state_cpu_report_sha256='statecheck',behavior_mu_cpu_report_sha256='mucheck',development_report_sha256='dev',direct_rank_record_sha256=ranks['record_sha256'])
    reviewer=write(gate/'external-review.json',review);write(gate/'independent-envelope.json',dict(review=reviewer,threshold_assessment=descriptor))
    return run,p,proofs,assessment,review

class V2Tests(unittest.TestCase):
    def test_foreign_same_id_suffix_legacy_unknown_and_own_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()/'intent.json';value=journal(path);write(path,value)
            kw=dict(cluster_name=s.CLUSTER,node_list=s.NODE,run_id='tree-fresh200-train-99',wait_seconds=0)
            self.assertEqual(c.submitted_intent(path,'m','r','99',**kw)['job_id'],99)
            cases=[dict(cluster_name='foreign'),dict(parsable_stdout='99;foreign',stdout='99;foreign\n'),dict(job_id=True),dict(status='UNKNOWN'),dict(run_id='other'),dict(node_list='h100-16'),dict(cluster='foreign')]
            for fields in cases:
                with self.subTest(fields=fields):
                    bad=dict(value,**fields);write(path,bad)
                    with self.assertRaises(ValueError):c.submitted_intent(path,'m','r','99',**kw)
            bad=value.copy();bad.pop('cluster_name');write(path,bad)
            with self.assertRaises(ValueError):c.submitted_intent(path,'m','r','99',**kw)
    def test_readonly_pending_finalization_wait_and_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()/'intent.json';value=journal(path);write(path,dict(value,status='SUBMITTING'));elapsed=[0.];calls=[]
            def sleep(t):elapsed[0]+=t;calls.append(t);write(path,value)
            kw=dict(cluster_name=s.CLUSTER,node_list=s.NODE,run_id='tree-fresh200-train-99',clock=lambda:elapsed[0],sleep=sleep)
            self.assertEqual(c.submitted_intent(path,'m','r','99',**kw)['status'],'SUBMITTED');self.assertEqual(calls,[.2])
            write(path,dict(value,status='SUBMITTING'))
            with self.assertRaises(ValueError):c.submitted_intent(path,'m','r','99',wait_seconds=0,**kw)
    def test_actual_runtime_queries_controlled_env_exact_identity(self):
        calls=[]
        def run(cmd,**kw):calls.append((cmd,kw));return scheduler(cmd,**kw)
        env={'SLURM_JOB_ID':'99','SLURM_JOB_NODELIST':'h100-10','SLURM_CLUSTER_NAME':s.CLUSTER,'SLURM_CONF':'foreign','SBATCH_CLUSTERS':'foreign','SCANCEL_CLUSTERS':'foreign'}
        observed=s.verify_runtime(env,'h100-10',run);self.assertEqual(observed['allocated_cpus'],96)
        self.assertTrue(all('--local' in cmd and not any(k in kw['env'] for k in s.EXCLUDED) and not any(k.startswith(s.PREFIXES) for k in kw['env']) for cmd,kw in calls))
        for field,bad in [('JobId','777'),('NodeList','h100-16'),('CPUs/Task','320'),('MinMemoryNode','128G'),('TimeLimit','UNLIMITED'),('Requeue','1'),('Restarts','1')]:
            value=ALLOCATION.replace(field+'='+dict(item.split('=',1) for item in ALLOCATION.split())[field],field+'='+bad)
            with self.subTest(field=field),self.assertRaises(ValueError):s.allocation_identity(value,'99',s.CLUSTER)
        with self.assertRaises(ValueError):s.allocation_identity(ALLOCATION+' JobId=777','99',s.CLUSTER)
    def test_foreign_local_config_runtime_declaration_federation_query_errors(self):
        for output in ['ClusterName = foreign\n','ClusterName = wuhan-h100\nClusterName = wuhan-h100\n']:
            with self.assertRaises(ValueError):s.cluster_identity(lambda *a,**k:N(returncode=0,stdout=output,stderr=''),{})
        with self.assertRaises(ValueError):s.verify_runtime({'SLURM_JOB_ID':'99','SLURM_JOB_NODELIST':s.NODE,'SLURM_CLUSTER_NAME':'foreign'},s.NODE,scheduler)
        with self.assertRaises(ValueError):s.cluster_identity(lambda cmd,**kw:N(returncode=0,stdout='ClusterName = wuhan-h100\n' if cmd[-1]=='config' else 'FederationName=f',stderr=''),{})
        with self.assertRaises(ValueError):s.cluster_identity(lambda *a,**kw:N(returncode=1,stdout='',stderr='accounting unreachable'),{})
    def test_guard_bare_cancel_is_never_executed_and_top_level_original_argv(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()/'refusal.json';calls=[];run=guard_entry.guarded_run(lambda *a,**k:calls.append((a,k)),path)
            with self.assertRaises(RuntimeError):run(['scancel','99'],check=False)
            self.assertEqual(calls,[]);self.assertFalse(json.loads(path.read_text())['scheduler_cancellation_executed'])
            run(['other','99'],env={'SLURM_CONF':'foreign','SBATCH_CLUSTERS':'foreign'})
            self.assertEqual(calls[0][1]['env'],{})
            tree=ast.parse((HERE/'guard_entry.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            assignments=[n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='argv' for t in n.targets)]
            execs=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='exec']
            self.assertTrue(assignments[0].lineno<execs[0].lineno);self.assertNotIn('module.main()',ast.unparse(main))
    def test_durable_initial_atomic_final_replace_unknown_and_exact_timeout_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()/'intent.json';value=dict(journal(path),status='SUBMITTING');calls=[]
            original=s.fsync_directory
            with patch.object(s,'fsync_directory',side_effect=lambda p:(calls.append(Path(p)),original(p))):
                s.create_journal(path,value);s.finalize_journal(path,dict(value,status='UNKNOWN'))
            self.assertEqual(calls,[path.parent,path.parent]);self.assertEqual(json.loads(path.read_text())['status'],'UNKNOWN')
            with self.assertRaises(FileExistsError):s.create_journal(path,value)
            payload=b'\x00\xffunknown\n';self.assertEqual(base64.b64decode(s.timeout_output(payload)['bytes']),payload)
            other=path.parent/'new.json';s.create_journal(other,value)
            with patch.object(s.os,'replace',side_effect=OSError('fixture power loss')):
                with self.assertRaises(OSError):s.finalize_journal(other,dict(value,status='SUBMITTED'))
            self.assertEqual(json.loads(other.read_text())['status'],'SUBMITTING')
    def test_no_duplicate_queue_and_no_foreign_or_unknown_prior_accounting(self):
        for text in ['99|tree-fresh200-train|RUNNING|gres/gpu:8\n','99|tree-fresh200-train|PENDING|N/A\n','unknown']:
            with self.assertRaises(ValueError):submitter.queue_clear(lambda *a,**kw:N(returncode=0,stdout=text,stderr=''),{})
        self.assertEqual(submitter.queue_clear(lambda *a,**kw:N(returncode=0,stdout='',stderr=''),{}),'')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();prior=root/'TREE_FRESH200_TRAIN_ONCE-old.json';target=root/'TREE_FRESH200_TRAIN_ONCE-new.json';value=journal(prior);write(prior,value)
            run=lambda *a,**kw:N(returncode=0,stdout='99|foreign|COMPLETED|0:0\n',stderr='')
            with self.assertRaises(ValueError):submitter.prior_journals_clear(target,run,{})
            run=lambda *a,**kw:N(returncode=0,stdout='99|wuhan-h100|COMPLETED|0:0\n',stderr='')
            self.assertEqual(len(submitter.prior_journals_clear(target,run,{})),1)
            write(prior,dict(value,status='UNKNOWN'))
            with self.assertRaises(ValueError):submitter.prior_journals_clear(target,run,{})
    def test_submit_result_foreign_ambiguous_suffix_rejected(self):
        for output in ['99;foreign\n','99;wuhan-h100;foreign\n','99\nextra','99_0\n']:
            with self.assertRaises(ValueError):submitter.parse_submission(N(returncode=0,stdout=output,stderr=''))
        self.assertEqual(submitter.parse_submission(N(returncode=0,stdout='99;wuhan-h100\n',stderr='')),(99,'99;wuhan-h100'))
    def test_zero_only_fullrun_threshold_receipt_is_rejected_but_bootstrap_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            p,_=observation(Path(temp).resolve());values={k:c.read_bound(v['path'],v['sha256']) for k,v in p['production_inputs'].items()}
            self.assertEqual(c.observed_production_inputs(p,values)['target_old_policy_kl'],.2)
            for fields in [dict(status='INDEPENDENT_FRESH200_MU_PRODUCTION_THRESHOLDS_ACCEPTED',production_thresholds_accepted=True),dict(zero_monitor_sha256={}),dict(zero_rank_observations=[]),dict(assessment_rationale=''),dict(continuation_6_200_accepted=True)]:
                bad=copy.deepcopy(values);bad['threshold_acceptance'][0].update(fields)
                with self.subTest(fields=fields),self.assertRaises(ValueError):c.observed_production_inputs(p,bad)
    def test_actual_dispatch_function_once_and_timeout_preserves_unknown(self):
        import hashlib
        import render_dispatch
        import subprocess
        for timeout in (False,True):
            with self.subTest(timeout=timeout),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();package=root/'instruct300/candidates/frozen';package.mkdir(parents=True)
                p,release_descriptor=observation(root/'observations')
                intent=root/'instruct300/TREE_FRESH200_TRAIN_ONCE-cpu-fixture.json'
                p.update(scope='FROZEN_PENDING_INDEPENDENT_FRESH200_TRAIN200_PRE_DISPATCH_REVIEW',stage='fresh_mu_train200',
                    bootstrap_thresholds_accepted=True,production_thresholds_accepted=False,dispatch={'once_intent':str(intent),'cluster_name':s.CLUSTER})
                for name in ('render_dispatch.py','submit_once.py'):(package/name).write_bytes((HERE/name).read_bytes())
                p['files']={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in ('render_dispatch.py','submit_once.py')}
                manifest=write(package/'manifest.json',p);release,_,_=c.read_bound(release_descriptor['path'])
                release.update(manifest_sha256=manifest['sha256'],dispatch_intent_path=str(intent),
                    dispatch_renderer_sha256=p['files']['render_dispatch.py'],dispatch_submitter_sha256=p['files']['submit_once.py'])
                release_descriptor=write(root/'release.json',release);calls=[]
                def run(command,**kw):
                    calls.append((command,kw))
                    if command[0]=='squeue':return N(returncode=0,stdout='',stderr='')
                    if command[0]=='sbatch':
                        if timeout:raise subprocess.TimeoutExpired(command,60,output=b'\x00\xffunconfirmed',stderr=b'\xfe')
                        return N(returncode=0,stdout='99;wuhan-h100\n',stderr='')
                    return scheduler(command,**kw)
                actual_inputs=c.production_inputs
                with patch.object(submitter,'ROOT',root),patch.object(render_dispatch,'ROOT',root),patch.object(submitter,'__file__',str(package/'submit_once.py')),patch.object(submitter,'production_inputs',side_effect=lambda *a,**kw:actual_inputs(*a,**kw,trusted_root=root)):
                    script=root/'dispatch.sbatch';script.write_text(render_dispatch.render(package,manifest['sha256'],release_descriptor['path'],release_descriptor['sha256'],72))
                    script_sha=hashlib.sha256(script.read_bytes()).hexdigest()
                    args=(package,manifest['sha256'],release_descriptor['path'],release_descriptor['sha256'],script,script_sha)
                    if timeout:
                        with self.assertRaises(subprocess.TimeoutExpired):submitter.submit(*args,run=run,hostname=s.NODE)
                    else:self.assertEqual(submitter.submit(*args,run=run,hostname=s.NODE)['job_id'],99)
                    value=json.loads(intent.read_text());self.assertEqual(value['status'],'UNKNOWN' if timeout else 'SUBMITTED')
                    if timeout:self.assertEqual(base64.b64decode(value['timeout_stdout']['bytes']),b'\x00\xffunconfirmed')
                    with self.assertRaises(ValueError):submitter.submit(*args,run=run,hostname=s.NODE)
                self.assertEqual(sum(command[0]=='sbatch' for command,kw in calls),1)
                self.assertTrue(all('-M' not in command and 'SLURM_CONF' not in kw['env'] for command,kw in calls))
    def test_v3_actual_path_toctou_cannot_change_verified_stdin(self):
        import hashlib
        import render_dispatch
        import subprocess
        for timeout in (False,True):
            with self.subTest(timeout=timeout),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();package=root/'instruct300/candidates/frozen';package.mkdir(parents=True)
                p,release_descriptor=observation(root/'observations')
                intent=root/'instruct300/TREE_FRESH200_TRAIN_ONCE-cpu-fixture.json'
                p.update(scope='FROZEN_PENDING_INDEPENDENT_FRESH200_TRAIN200_PRE_DISPATCH_REVIEW',stage='fresh_mu_train200',
                    bootstrap_thresholds_accepted=True,production_thresholds_accepted=False,dispatch={'once_intent':str(intent),'cluster_name':s.CLUSTER})
                for name in ('render_dispatch.py','submit_once.py'):(package/name).write_bytes((HERE/name).read_bytes())
                p['files']={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in ('render_dispatch.py','submit_once.py')}
                manifest=write(package/'manifest.json',p);release,_,_=c.read_bound(release_descriptor['path'])
                release.update(manifest_sha256=manifest['sha256'],dispatch_intent_path=str(intent),
                    dispatch_renderer_sha256=p['files']['render_dispatch.py'],dispatch_submitter_sha256=p['files']['submit_once.py'])
                release_descriptor=write(root/'release.json',release);calls=[]
                def run(command,**kw):
                    calls.append((command,kw))
                    if command[0]=='squeue':
                        script.write_bytes(script.read_bytes().replace(b'--time=72:00:00',b'--time=999:00:00'))
                        return N(returncode=0,stdout='',stderr='')
                    if command[0]=='sbatch':
                        self.assertEqual(command,['sbatch','--parsable'])
                        self.assertEqual(kw['input'],reviewed_bytes)
                        self.assertNotIn('text',kw)
                        self.assertNotEqual(script.read_bytes(),reviewed_bytes)
                        if timeout:raise subprocess.TimeoutExpired(command,60,output=b'\x00\xffunconfirmed',stderr=b'\xfe')
                        return N(returncode=0,stdout=b'99;wuhan-h100\n',stderr=b'')
                    return scheduler(command,**kw)
                actual_inputs=c.production_inputs
                with patch.object(submitter,'ROOT',root),patch.object(render_dispatch,'ROOT',root),patch.object(submitter,'__file__',str(package/'submit_once.py')),patch.object(submitter,'production_inputs',side_effect=lambda *a,**kw:actual_inputs(*a,**kw,trusted_root=root)):
                    script=root/'dispatch.sbatch';script.write_text(render_dispatch.render(package,manifest['sha256'],release_descriptor['path'],release_descriptor['sha256'],72))
                    reviewed_bytes=script.read_bytes();script_sha=hashlib.sha256(reviewed_bytes).hexdigest()
                    args=(package,manifest['sha256'],release_descriptor['path'],release_descriptor['sha256'],script,script_sha)
                    if timeout:
                        with self.assertRaises(subprocess.TimeoutExpired):submitter.submit(*args,run=run,hostname=s.NODE)
                    else:self.assertEqual(submitter.submit(*args,run=run,hostname=s.NODE)['job_id'],99)
                    value=json.loads(intent.read_text());self.assertEqual(value['status'],'UNKNOWN' if timeout else 'SUBMITTED')
                    self.assertEqual(base64.b64decode(value['submitted_input_base64']),reviewed_bytes)
                    self.assertEqual(value['submitted_input_sha256'],script_sha)
                    self.assertEqual(value['submitted_input_bytes'],len(reviewed_bytes))
                    self.assertEqual(value['submit_argv'],['sbatch','--parsable'])
                    self.assertEqual(value['submit_mode'],'verified_bytes_via_stdin')
                    if timeout:self.assertEqual(base64.b64decode(value['timeout_stdout']['bytes']),b'\x00\xffunconfirmed')
                    with self.assertRaises(ValueError):submitter.submit(*args,run=run,hostname=s.NODE)
                self.assertEqual(sum(command[0]=='sbatch' for command,kw in calls),1)
                self.assertTrue(all('-M' not in command and 'SLURM_CONF' not in kw['env'] for command,kw in calls))
    def test_v3_invalid_utf8_or_nul_script_never_reaches_submission(self):
        import hashlib
        import render_dispatch
        import subprocess
        for invalid_suffix in (b'\xff',b'\x00'):
            timeout=False
            with self.subTest(invalid_suffix=invalid_suffix),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();package=root/'instruct300/candidates/frozen';package.mkdir(parents=True)
                p,release_descriptor=observation(root/'observations')
                intent=root/'instruct300/TREE_FRESH200_TRAIN_ONCE-cpu-fixture.json'
                p.update(scope='FROZEN_PENDING_INDEPENDENT_FRESH200_TRAIN200_PRE_DISPATCH_REVIEW',stage='fresh_mu_train200',
                    bootstrap_thresholds_accepted=True,production_thresholds_accepted=False,dispatch={'once_intent':str(intent),'cluster_name':s.CLUSTER})
                for name in ('render_dispatch.py','submit_once.py'):(package/name).write_bytes((HERE/name).read_bytes())
                p['files']={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in ('render_dispatch.py','submit_once.py')}
                manifest=write(package/'manifest.json',p);release,_,_=c.read_bound(release_descriptor['path'])
                release.update(manifest_sha256=manifest['sha256'],dispatch_intent_path=str(intent),
                    dispatch_renderer_sha256=p['files']['render_dispatch.py'],dispatch_submitter_sha256=p['files']['submit_once.py'])
                release_descriptor=write(root/'release.json',release);calls=[]
                def run(command,**kw):
                    calls.append((command,kw))
                    if command[0]=='squeue':return N(returncode=0,stdout='',stderr='')
                    if command[0]=='sbatch':
                        if timeout:raise subprocess.TimeoutExpired(command,60,output=b'\x00\xffunconfirmed',stderr=b'\xfe')
                        return N(returncode=0,stdout='99;wuhan-h100\n',stderr='')
                    return scheduler(command,**kw)
                actual_inputs=c.production_inputs
                with patch.object(submitter,'ROOT',root),patch.object(render_dispatch,'ROOT',root),patch.object(submitter,'__file__',str(package/'submit_once.py')),patch.object(submitter,'production_inputs',side_effect=lambda *a,**kw:actual_inputs(*a,**kw,trusted_root=root)):
                    script=root/'dispatch.sbatch';script.write_text(render_dispatch.render(package,manifest['sha256'],release_descriptor['path'],release_descriptor['sha256'],72))
                    script.write_bytes(script.read_bytes()+invalid_suffix)
                    script_sha=hashlib.sha256(script.read_bytes()).hexdigest()
                    args=(package,manifest['sha256'],release_descriptor['path'],release_descriptor['sha256'],script,script_sha)
                    with self.assertRaises(ValueError):submitter.submit(*args,run=run,hostname=s.NODE)
                    self.assertFalse(intent.exists())
                self.assertEqual(sum(command[0]=='sbatch' for command,kw in calls),0)
    def test_actual_first5_original_review_bytes_need_short_assessment(self):
        with tempfile.TemporaryDirectory() as temp:
            run,p,proofs,assessment,review=five_fixture(Path(temp).resolve());gate=run/'output/acceptance-step5'
            result=c.publish_external_approval(run,p,'config',proofs)
            self.assertEqual((gate/'approved.json').read_bytes(),(gate/'external-review.json').read_bytes());self.assertTrue(result['full_run_thresholds_accepted'])
    def test_existing_exact_f7_source_none_receipt_is_source_only(self):
        with tempfile.TemporaryDirectory() as temp:
            p,_=observation(Path(temp).resolve());values={k:c.read_bound(v['path'],v['sha256']) for k,v in p['production_inputs'].items()}
            frozen=c.read_bound(HERE.parent/'tree-fresh200-f7-v8-v6-context-free-review/RECEIPT.json','6f2ceaebd1ff6c04e9330b404e17ff7db9df7493db5c686cbc3366cb3253fb44')
            p['code_commit']=frozen[0]['source']['audited_clean_head']
            values['actual_calibration_ready'][0]['code_commit']=p['code_commit'];values['threshold_acceptance'][0]['code_commit']=p['code_commit']
            values['source_context_free_review']=frozen
            self.assertEqual(c.observed_production_inputs(p,values)['target_old_policy_kl'],.2)
            self.assertFalse(frozen[0]['actual_gpu_accepted']);self.assertFalse(frozen[0]['production_thresholds_accepted'])
            values['source_context_free_review']=(frozen[0],'0'*64,frozen[2])
            with self.assertRaises(ValueError):c.observed_production_inputs(p,values)
    def test_first5_no_threshold_assessment_unknown_wrong_job_and_values_never_publish(self):
        for mutation in ['missing','wrong_job','changed_threshold','fake_observations','empty_rationale','different_partial_proof']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temp:
                run,p,proofs,assessment,review=five_fixture(Path(temp).resolve());gate=run/'output/acceptance-step5'
                if mutation=='missing':write(gate/'independent-envelope.json',{'review':write(gate/'external-review.json',review)})
                else:
                    if mutation=='wrong_job':assessment['job_id']='100'
                    if mutation=='changed_threshold':assessment['values']['negative_tail_limit']=.2
                    if mutation=='fake_observations':assessment['observations']['max_sampled_old_policy_kl']=0
                    if mutation=='empty_rationale':assessment['assessment_rationale']=' '
                    if mutation=='different_partial_proof':assessment['partial_raw_cpu_report_sha256']='other'
                    descriptor=write(gate/'external-threshold-assessment.json',assessment);review['threshold_assessment_sha256']=descriptor['sha256']
                    write(gate/'independent-envelope.json',dict(review=write(gate/'external-review.json',review),threshold_assessment=descriptor))
                with self.assertRaises((ValueError,KeyError)):c.publish_external_approval(run,p,'config',proofs)
                self.assertFalse((gate/'approved.json').exists())
    def test_missing_real_short_monitor_or_changed_actual_monitor_refuses(self):
        with tempfile.TemporaryDirectory() as temp:
            run,p,proofs,assessment,review=five_fixture(Path(temp).resolve())
            path=next(iter(proofs['tree_and_safety']['evidence_sha256']));Path(path).unlink()
            with self.assertRaises(FileNotFoundError):c.publish_external_approval(run,p,'config',proofs)
    def test_missing_raw_worker_driver_actor_and_changed_bytes_block_actual_accounting(self):
        import attempt_accounting as accounting
        for name in ('worker-00.pt','worker-07.pt','driver.pt','actor-inputs.pt','changed'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                run=Path(temp).resolve()/'tree-fresh200-train-99';allskip(run,1,0);direct(run,1,attempt=2)
                observed=accounting.attempt_results(run,package(),'config',1)
                event=json.loads((run/'checkpoints/training_state/attempt-results/attempt-000001.json').read_text())
                entry=dict(event['partial_evidence'],attempt_id=1,credited_outer_candidate=False,optimizer_calls_per_rank={str(r):0 for r in range(8)})
                saved=dict(protocol=accounting.PROTOCOL,completed_outer_updates=1,training_attempts=2,refused_attempts=1,guard_refused_attempts=1,
                    completed_optimizer_calls_per_rank=observed['completed_optimizer_calls_per_rank'],total_optimizer_calls_per_rank=observed['total_optimizer_calls_per_rank'],
                    partial_optimizer_calls_per_rank=observed['partial_optimizer_calls_per_rank'],partial_evidence=[entry])
                directory=run/'checkpoints/training_state/partial_attempt_000001_outer_000001'
                if name=='changed':(directory/'worker-03.pt').write_bytes(b'mutated')
                else:(directory/name).unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):accounting.attempt_results(run,package(),'config',1)
                with self.assertRaises((ValueError,FileNotFoundError)):accounting.checkpoint_accounting(saved,observed,1,run)
    def test_serial_actual_partial_report_empty_and_forged_coverage_rejected(self):
        import train_monitor as monitor
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp).resolve()/'tree-fresh200-train-99';checkpoint(run,5);(run/'output').mkdir(exist_ok=True)
            owned=N(spawn=lambda *a,**kw:N(poll=lambda:0,returncode=0))
            queue=monitor.CpuChecks(owned,run,HERE,package(),'config',5);queue.complete=True
            write(queue.directory/'behavior-mu.json',dict(status='FRESH200_REAL_BEHAVIOR_MU_CPU_BINDING_VERIFIED',completed_outers_checked=5,real_DataProto=True,
                  optimizer_writes=0,model_forwards=0,cuda_initialized=False,original_sampler_actor_gate_accepted=False))
            value=dict(status='FRESH200_PARTIAL_RAW_CPU_REVERIFIED',identity=identity(run),completed_outer=5,checkpoint_manifest_sha256=queue.manifest_sha,
                  same_fd_io=True,optimizer_writes=0,model_forwards=0,cuda_initialized=False,raw_files_sha256={},raw_files_checked=0,
                  partial_attempts_checked=0,logical_records=[],pt_deserialization_count=0,actual_pt_deserialization=False)
            write(queue.directory/'partial-raw.json',value)
            with patch.object(monitor,'hf_reload_report',return_value='synthetic'),patch.object(monitor,'state_cpu_report',return_value='synthetic'),patch.object(monitor,'tree_and_safety_records',return_value={}),patch.object(monitor,'training_cost_reports',return_value={}),patch.object(monitor,'validation_reports',return_value={}):
                self.assertEqual(queue.proofs()['partial_raw_files_sha256'],{})
                for fields in ({'actual_pt_deserialization':True},{'pt_deserialization_count':10},{'raw_files_sha256':{'missing':'f'*64}},{'checkpoint_manifest_sha256':'wrong'}):
                    with self.subTest(fields=fields):
                        write(queue.directory/'partial-raw.json',dict(value,**fields))
                        with self.assertRaises(ValueError):queue.proofs()

if __name__=='__main__':unittest.main()
