"""Independent exact V6 source functions + isolated CPU mocks, no Slurm calls."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as N
from unittest.mock import patch

BASE=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
HELP=BASE/'work/tree-fresh200-dispatch-v6'
PREVIOUS=BASE/'work/tree-fresh200-dispatch-v5'
V5REPORT=BASE/'work/tree-fresh200-f7-v8-v5-context-free-review/wrapper-helper-independent'
WRAP=BASE/'work/tree-fresh200-mu-runtime-v8'
SOURCE=BASE/'work/tree-behavior-mu-ppo'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def record(path):
    raw=path.read_bytes();return dict(path=str(path),sha256=sha(raw),bytes=len(raw))
def save(name,value):
    path=OUT/name;path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');return record(path)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
sys.dont_write_bytecode=True
assert sys.flags.optimize==0
C=load('v6_independent_common',HELP/'fresh200_dispatch_common.py')
S=load('v6_independent_submit',HELP/'submit_tree_fresh200_once.py')
pins={HELP/'FILE_HASHES_V6.json':'88df8b11faff91f6d30adc4625d3fbf370686746de6e6036bfb4812692439099',
      HELP/'SOURCE_CPU_RECEIPT_V6.json':'1e20539f1279d4b92d2fd9321d6cb8bf9fabda125bdd3ff74d4c6a9d848af495',
      HELP/'submit_tree_fresh200_once.py':'8f180d80d5128f36cc414dc1aef005ab3e1456dbfda0710fa822486344dd8d26'}
for path,expected in pins.items():assert record(path)['sha256']==expected,path
initial_v5={str(path):record(path) for path in V5REPORT.iterdir() if path.is_file()}
inventory=json.loads((HELP/'FILE_HASHES_V6.json').read_bytes())
for name,expected in inventory['files'].items():
    observed=record(HELP/name)
    assert {key:observed[key] for key in ['sha256','bytes']}==expected,name
helpers={name:record(HELP/name) for name in ['fresh200_dispatch_common.py','prepare_tree_fresh200_package.py','submit_tree_fresh200_once.py']}
for name in ['fresh200_dispatch_common.py','prepare_tree_fresh200_package.py']:
    assert (HELP/name).read_bytes()==(PREVIOUS/name).read_bytes(),name
oldmap=json.loads((V5REPORT/'BYTE_MAPS.json').read_bytes())
wrapper_map={name:dict(record(WRAP/name),matches=True) for name in C.WRAPPER_FILES}
for name,entry in wrapper_map.items():assert entry['sha256']==oldmap['wrapper_files'][name]['sha256']
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
clean=not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip()
assert head==C.COMMIT and clean
save('BYTE_AUDIT.json',dict(git_head=head,source_clean=clean,helper_files=helpers,
    wrapper_files=wrapper_map,runtime_entries=oldmap['runtime_entries'],
    common_and_prepare_exactly_unchanged_from_v5=True,v6_inventory_all_records_verified=True,
    pins={str(path):record(path) for path in pins},v5_review_artifacts_before=initial_v5,
    actual_gpu_executed=False,actual_remote_executed=False,actual_slurm_executed=False))

malicious={key:'synthetic-foreign' for key in ['SBATCH_CLUSTERS','SBATCH_NODES','SBATCH_EXTRA',
 'SQUEUE_CLUSTERS','SQUEUE_FEDERATION','SACCT_CLUSTERS','SACCT_FEDERATION',
 'SCONTROL_CLUSTERS','SCONTROL_SIBLING','SLURM_CLUSTERS','SLURM_CONF','SLURM_CONF_SERVER',
 'SLURM_CLUSTER_NAME','SLURM_FEDERATION']}
malicious['INDEPENDENT_V6_ENV_SENTINEL']='keep-this-synthetic-value'
results=[]

def case(name,*,stdout='777',exception=None,config='ClusterName = wuhan-h100\n',
         federation='',config_stderr='',config_rc=0,fed_stderr='',fed_rc=0,
         prior=None,acct='777|wuhan-h100|COMPLETED|0:0\n',atomic_fault=False,initial_dir_fault=False,
         final_dir_fault=False,trace_durability=False,expect='SUBMITTED'):
    with tempfile.TemporaryDirectory(dir=OUT,prefix='v6-case-') as tmp:
        root=Path(tmp);(root/'instruct300/candidates').mkdir(parents=True)
        target=root/'instruct300/candidates/fixture';target.mkdir()
        calls=[];events=[];journals=[]
        journal=root/'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-aaaaaaaaaaaa.json'
        if prior is not None:
            p=root/'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-cccccccccccc.json'
            p.write_text(json.dumps(prior))
        def run(command,**kwargs):
            env=kwargs['env']
            assert not (set(malicious)-{'INDEPENDENT_V6_ENV_SENTINEL'})&set(env)
            assert env['INDEPENDENT_V6_ENV_SENTINEL']=='keep-this-synthetic-value'
            calls.append(dict(command=command,env_sanitized=True,remaining_synthetic_sentinel=True))
            if command[0]=='scontrol':
                assert '--local' in command
                if command[-1]=='config':return N(returncode=config_rc,stdout=config,stderr=config_stderr)
                assert command[-1]=='federation';return N(returncode=fed_rc,stdout=federation,stderr=fed_stderr)
            if command[0]=='squeue':
                assert '--local' in command and '-w' in command and 'h100-9' in command
                return N(returncode=0,stdout='',stderr='')
            if command[0]=='sacct':
                assert '--local' in command and 'JobIDRaw,Cluster,State,ExitCode' in command
                return N(returncode=0,stdout=acct,stderr='')
            assert command[0]=='sbatch' and '--nodelist=h100-9' in command
            initial=json.loads(journal.read_bytes())
            assert initial['status']=='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN' and initial['cluster_name']=='wuhan-h100'
            events.append('sbatch_mock_after_initial_intent');journals.append(dict(at='before_sbatch',value=initial))
            if exception is not None:raise exception
            return N(returncode=0,stdout=stdout,stderr='')
        true_fsync=S.os.fsync;true_replace=S.os.replace;true_dir=S.fsync_directory
        def fsync(fd):
            events.append('fsync_directory_fd' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'fsync_file_fd')
            return true_fsync(fd)
        def replace(before,after):
            events.append('atomic_replace')
            if atomic_fault:raise OSError('independent atomic replace fault')
            return true_replace(before,after)
        directory_calls=[0]
        def directory(path):
            directory_calls[0]+=1
            if initial_dir_fault and directory_calls[0]==1:raise OSError('independent initial directory fsync fault')
            if final_dir_fault and directory_calls[0]==2:raise OSError('independent final directory fsync fault')
            return true_dir(path)
        with patch.object(C,'ROOT',root),patch.object(C,'protected_processes',lambda:None),\
             patch.dict(os.environ,malicious),patch.object(S.os,'fsync',side_effect=fsync),\
             patch.object(S.os,'replace',side_effect=replace),patch.object(S,'fsync_directory',side_effect=directory):
            try:value=S.submit_once(C,target,'a'*64,'b'*64,{},run=run);error=None
            except Exception as exc:value=None;error=repr(exc)
            final=json.loads(journal.read_bytes()) if journal.exists() else None
            if expect=='SUBMITTED':
                assert error is None and value['status']=='SUBMITTED'
            elif expect=='REJECT_BEFORE_SBATCH':
                assert error and not any(call['command'][0]=='sbatch' for call in calls) and final is None
            elif expect=='UNKNOWN':
                assert error and final['status']=='SUBMISSION_UNKNOWN_DO_NOT_RETRY' and 'job_id' not in final
            elif expect=='INITIAL_BLOCKING':
                assert error and final['status']=='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN'
            elif expect=='FINAL_BLOCKING':
                assert error and final['status']=='SUBMITTED'
            else:raise AssertionError(expect)
            calls_before_retry=len(calls)
            if final is not None:
                try:S.submit_once(C,target,'a'*64,'b'*64,{},run=run)
                except Exception as exc:retry_error=repr(exc)
                else:raise AssertionError('same intent unexpectedly retried')
                assert not any(call['command'][0]=='sbatch' for call in calls[calls_before_retry:])
            else:retry_error=None
            second_calls_before=len(calls)
            if final is not None and final['status']!='SUBMITTED':
                try:S.submit_once(C,target,'e'*64,'b'*64,{},run=run)
                except Exception as exc:cross_error=repr(exc)
                else:raise AssertionError('unknown prior unexpectedly permitted next intent')
                assert not any(call['command'][0]=='sbatch' for call in calls[second_calls_before:])
            else:cross_error=None
        if trace_durability:
            expected=['fsync_file_fd','fsync_directory_fd','sbatch_mock_after_initial_intent',
                      'fsync_file_fd','atomic_replace','fsync_directory_fd']
            assert events==expected,events
        if isinstance(exception,subprocess.TimeoutExpired):
            assert base64.b64decode(final['exception_stdout']['data'])==exception.output
            assert base64.b64decode(final['exception_stderr']['data'])==exception.stderr
        results.append(dict(name=name,expect=expect,actual_result=value,error=error,
            final_journal=final,initial_journal_snapshots=journals,command_calls=calls,
            same_intent_retry_error=retry_error,unknown_cross_intent_retry_error=cross_error,
            syscall_order=events,temp_final_file_count=len(list((root/'instruct300').glob('*.final-*')))))

case('H01_foreign_suffix_rejected_and_both_intents_blocked',stdout='777;other_cluster',expect='UNKNOWN')
case('same_cluster_qualified_suffix_is_legal',stdout='777;wuhan-h100',trace_durability=True)
case('bare_local_id_is_legal',stdout='777',trace_durability=True)
for label,prior in [('missing',{'status':'SUBMITTED','job_id':777}),
                    ('foreign',{'status':'SUBMITTED','job_id':777,'cluster_name':'other_cluster'})]:
    case('H01_'+label+'_prior_cluster_rejected',prior=prior,expect='REJECT_BEFORE_SBATCH')
prior=dict(status='SUBMITTED',job_id=777,cluster_name='wuhan-h100')
case('same_cluster_prior_terminal_permits_new_intent',prior=prior)
for label,acct in [('foreign_cluster','777|other_cluster|COMPLETED|0:0\n'),
                   ('missing_cluster','777|COMPLETED|0:0\n'),('wrong_job','778|wuhan-h100|COMPLETED|0:0\n'),
                   ('running','777|wuhan-h100|RUNNING|0:0\n'),('empty',''),
                   ('duplicate','777|wuhan-h100|COMPLETED|0:0\n777|wuhan-h100|COMPLETED|0:0\n')]:
    case('accounting_'+label+'_rejects_before_submit',prior=prior,acct=acct,expect='REJECT_BEFORE_SBATCH')
for label,options in [('foreign_cluster',dict(config='ClusterName = foreign\n')),
 ('duplicate_cluster',dict(config='ClusterName = wuhan-h100\nClusterName = wuhan-h100\n')),
 ('missing_cluster',dict(config='NoClusterNameHere\n')),('config_stderr',dict(config_stderr='error')),
 ('config_nonzero',dict(config_rc=1)),('federation_present',dict(federation='FederationName = foreign')),
 ('federation_stderr',dict(fed_stderr='error')),('federation_nonzero',dict(fed_rc=1))]:
    case('cluster_probe_'+label+'_rejects_before_intent',**options,expect='REJECT_BEFORE_SBATCH')
case('TimeoutExpired_non_utf8_partial_bytes_preserved',exception=subprocess.TimeoutExpired(['synthetic-sbatch'],60,
     output=b'777;wuhan-h100\xff',stderr=b'partial\x00'),expect='UNKNOWN')
case('atomic_replace_failure_preserves_durable_initial_intent',atomic_fault=True,expect='INITIAL_BLOCKING')
case('initial_directory_fsync_failure_never_calls_sbatch',initial_dir_fault=True,expect='INITIAL_BLOCKING')
case('final_directory_fsync_failure_keeps_blocking_final_record',final_dir_fault=True,expect='FINAL_BLOCKING')
save('INDEPENDENT_SOURCE_FUNCTION_CASES.json',dict(status='ALL_SOURCE_FUNCTION_CASES_MATCHED_EXPECTATIONS',
    case_count=len(results),cases=results,actual_scheduler_commands_executed=0,
    actual_remote_executed=False,actual_gpu_executed=False,actual_assets_loaded=False,
    fixtures_only=True,source_helper_sha256=helpers['submit_tree_fresh200_once.py']['sha256']))

affected=['test_foreign_cluster_suffix_is_unknown_and_blocks_new_intent','test_same_cluster_qualified_id_is_preserved',
 'test_all_slurm_calls_strip_cluster_config_and_federation_env','test_wrong_local_cluster_or_configured_federation_prevents_submit',
 'test_missing_or_foreign_prior_cluster_refuses_same_number_alias','test_accounting_same_number_foreign_cluster_cannot_be_terminal',
 'test_timeout_preserves_exact_partial_bytes_without_retry','test_failed_atomic_final_replace_retains_initial_intent',
 'test_initial_directory_fsync_failure_never_calls_submit']
command=[sys.executable,'-B','-m','unittest','-v',*['test_fresh200_dispatch_cpu.Tests.'+name for name in affected]]
start=time.monotonic();test=subprocess.run(command,cwd=HELP,text=True,capture_output=True,timeout=60,
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',CUDA_VISIBLE_DEVICES=''))
elapsed=time.monotonic()-start
(OUT/'AFFECTED_CPU_ACTUAL.stdout').write_text(test.stdout);(OUT/'AFFECTED_CPU_ACTUAL.stderr').write_text(test.stderr)
assert test.returncode==0,test.stderr
save('AFFECTED_CPU_TEST_RESULTS.json',dict(command=command,cwd=str(HELP),unique_method_count=len(affected),
    actual_exit_code=test.returncode,actual_seconds=elapsed,
    stdout=record(OUT/'AFFECTED_CPU_ACTUAL.stdout'),stderr=record(OUT/'AFFECTED_CPU_ACTUAL.stderr'),
    historical_v5_methods_not_recounted=True,actual_slurm_executed=False,actual_gpu_executed=False,
    actual_remote_executed=False,actual_model_or_dataset_assets_loaded=False))
assert initial_v5=={str(path):record(path) for path in V5REPORT.iterdir() if path.is_file()}
for path,expected in pins.items():assert record(path)['sha256']==expected
for name,entry in helpers.items():assert record(HELP/name)==entry
for name,entry in wrapper_map.items():
    assert record(WRAP/name)=={key:value for key,value in entry.items() if key!='matches'}
save('IMMUTABILITY_AFTER_CHECKS.json',dict(v5_review_all_original_files_unchanged=True,
    v6_pinned_inventory_owner_and_submit_unchanged=True,v6_helper3_unchanged=True,
    wrapper15_unchanged=True,actual_remote_executed=False,actual_gpu_executed=False,actual_slurm_executed=False))
print(json.dumps(dict(status='INDEPENDENT_V6_CHECKS_PASSED',source_function_case_count=len(results),
    affected_unique_unittest_method_count=len(affected),helper_files=helpers,
    actual_remote_executed=False,actual_gpu_executed=False,actual_slurm_executed=False),indent=2))
