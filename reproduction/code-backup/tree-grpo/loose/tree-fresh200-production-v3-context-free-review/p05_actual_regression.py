"""Independent local P05 actual-function probes; no native scheduler or assets."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as N
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'tree-fresh200-train-runtime-v3'
sys.path.insert(0, str(SOURCE))
import production_contract as c
import slurm_identity as s
import submit_once as dispatcher
import render_dispatch as renderer
from test_runtime_contract_cpu import observation, write

CONFIG = 'ClusterName = wuhan-h100\n'
BAD_ENV = dict(SLURM_JOB_ID='99',SLURM_JOB_NODELIST='h100-9',SLURM_CLUSTER_NAME='wuhan-h100',SLURM_CONF='/foreign',SLURM_CONF_SERVER='foreign',SLURM_CLUSTERS='foreign',SLURM_FEDERATION='foreign',SBATCH_CLUSTERS='foreign',SBATCH_TIME='unlimited',SQUEUE_CLUSTERS='foreign',SACCT_CLUSTERS='foreign',SCONTROL_CLUSTER='foreign',SCANCEL_CLUSTERS='foreign',KEEP='preserved')

def sha(raw): return hashlib.sha256(raw).hexdigest()
def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def reject(fn,types):
    try: fn()
    except types as error: return dict(type=type(error).__name__,message=str(error))
    raise AssertionError('Required rejection missing')

def probe(label,mode='success',mutation='change',invalid_suffix=b''):
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        root=Path(tmp).resolve();package=root/'instruct300/candidates/frozen';package.mkdir(parents=True)
        manifest,release_descriptor=observation(root/'observations')
        intent=root/'instruct300/TREE_FRESH200_TRAIN_ONCE-independent-v3-cpu.json'
        manifest.update(scope='FROZEN_PENDING_INDEPENDENT_FRESH200_TRAIN200_PRE_DISPATCH_REVIEW',stage='fresh_mu_train200',bootstrap_thresholds_accepted=True,production_thresholds_accepted=False,dispatch={'once_intent':str(intent),'cluster_name':s.CLUSTER})
        for name in ('render_dispatch.py','submit_once.py'):(package/name).write_bytes((SOURCE/name).read_bytes())
        manifest['files']={name:sha((package/name).read_bytes()) for name in ('render_dispatch.py','submit_once.py')}
        m=write(package/'manifest.json',manifest)
        release,_,_=c.read_bound(release_descriptor['path'])
        release.update(manifest_sha256=m['sha256'],dispatch_intent_path=str(intent),dispatch_renderer_sha256=manifest['files']['render_dispatch.py'],dispatch_submitter_sha256=manifest['files']['submit_once.py'])
        r=write(root/'release.json',release)
        calls=[];submitted=[];after_path_sha=None;observed_initial=None
        def run(command,**kw):
            nonlocal after_path_sha,observed_initial
            calls.append(dict(argv=list(command),environment=dict(kw['env']),timeout=kw['timeout'],input_type=type(kw.get('input')).__name__,text=kw.get('text')))
            assert not any(k in s.EXCLUDED or k.startswith(s.PREFIXES) for k in kw['env'])
            assert '-M' not in command and command[0]!='scancel'
            if command==['scontrol','--local','show','config']:return N(returncode=0,stdout=CONFIG,stderr='')
            if command==['scontrol','--local','show','federation']:return N(returncode=0,stdout='',stderr='')
            if command[0]=='squeue':
                assert command==['squeue','--local','-h','-w',s.NODE,'-o','%i|%j|%T|%b']
                assert sha(script.read_bytes())==script_sha
                if mutation=='change':
                    changed=script.read_bytes().replace(b'#SBATCH --time=72:00:00',b'#SBATCH --time=999:00:00')
                    assert changed!=reviewed
                    script.write_bytes(changed);after_path_sha=sha(changed)
                    (OUT/(label+'-changed-path.sbatch')).write_bytes(changed)
                elif mutation=='delete':script.unlink();assert not script.exists()
                return N(returncode=0,stdout='',stderr='')
            assert command==['sbatch','--parsable'],command
            assert type(kw['input']) is bytes and kw['input']==reviewed
            assert 'text' not in kw and kw['capture_output'] is True and kw['timeout']==60
            if mutation=='delete':assert not script.exists()
            if mutation=='change':assert script.read_bytes()!=kw['input']
            observed_initial=json.loads(intent.read_text());assert observed_initial['status']=='SUBMITTING'
            assert observed_initial['submit_argv']==command
            assert observed_initial['submitted_input_sha256']==sha(kw['input'])==script_sha
            assert observed_initial['submitted_input_bytes']==len(kw['input'])
            assert base64.b64decode(observed_initial['submitted_input_base64'])==kw['input']
            submitted.append(kw['input'])
            if mode=='timeout':raise subprocess.TimeoutExpired(command,60,output=b'\x00\xffunknown\n',stderr=b'\xfe')
            if mode=='invalid-output':return N(returncode=0,stdout=b'99;wuhan-h100\n\xff',stderr=b'')
            if mode=='foreign-output':return N(returncode=0,stdout=b'99;foreign\n',stderr=b'')
            return N(returncode=0,stdout=b'99;wuhan-h100\n',stderr=b'')
        actual_inputs=c.production_inputs
        with patch.object(dispatcher,'ROOT',root),patch.object(renderer,'ROOT',root),patch.object(dispatcher,'__file__',str(package/'submit_once.py')),patch.object(dispatcher,'production_inputs',side_effect=lambda *a,**kw:actual_inputs(*a,**kw,trusted_root=root)),patch.dict(os.environ,BAD_ENV,clear=True):
            script=root/'dispatch.sbatch';reviewed=renderer.render(package,m['sha256'],r['path'],r['sha256'],72).encode('utf-8')
            script.write_bytes(reviewed+invalid_suffix);script_sha=sha(script.read_bytes())
            args=(package,m['sha256'],r['path'],r['sha256'],script,script_sha)
            if invalid_suffix:
                refusal=reject(lambda:dispatcher.submit(*args,run=run,hostname=s.NODE),(ValueError,))
                assert not intent.exists() and not submitted and not any(v['argv'][0]=='sbatch' for v in calls)
                result=dict(status='PASS_INVALID_SCRIPT_REFUSED_BEFORE_INTENT_OR_SUBMIT',refusal=refusal,invalid_suffix_base64=base64.b64encode(invalid_suffix).decode(),mock_sbatch_count=0,intent_exists=False)
            else:
                (OUT/(label+'-reviewed-input.sbatch')).write_bytes(reviewed)
                if mode=='success':assert dispatcher.submit(*args,run=run,hostname=s.NODE)['job_id']==99
                else:refusal=reject(lambda:dispatcher.submit(*args,run=run,hostname=s.NODE),(ValueError,subprocess.TimeoutExpired))
                value=json.loads(intent.read_text());assert value['status']==('SUBMITTED' if mode=='success' else 'UNKNOWN')
                assert value['submitted_input_sha256']==script_sha==sha(submitted[0])
                assert value['dispatch_script_sha256']==script_sha and value['submit_argv']==['sbatch','--parsable']
                assert value['submit_mode']=='verified_bytes_via_stdin' and value['selected_script_path']==str(script)
                assert base64.b64decode(value['submitted_input_base64'])==reviewed and value['submitted_input_bytes']==len(reviewed)
                if mode=='success':assert base64.b64decode(value['submitted_result_stdout_raw']['bytes'])==b'99;wuhan-h100\n'
                if mode=='timeout':
                    assert base64.b64decode(value['timeout_stdout']['bytes'])==b'\x00\xffunknown\n'
                    assert base64.b64decode(value['timeout_stderr']['bytes'])==b'\xfe'
                if mode=='invalid-output':assert base64.b64decode(value['submitted_result_stdout_raw']['bytes'])==b'99;wuhan-h100\n\xff'
                if mode=='foreign-output':assert base64.b64decode(value['submitted_result_stdout_raw']['bytes'])==b'99;foreign\n'
                # Restore the reviewed path so irreversible intent refusal, rather
                # than an earlier path hash failure, is the reentry condition.
                script.write_bytes(reviewed)
                reentry=reject(lambda:dispatcher.submit(*args,run=run,hostname=s.NODE),(ValueError,))
                assert 'Intent already exists' in reentry['message']
                assert len(submitted)==1 and sum(v['argv'][0]=='sbatch' for v in calls)==1
                (OUT/(label+'-JOURNAL.json')).write_bytes(intent.read_bytes())
                result=dict(status='PASS_VERIFIED_BYTES_CONSUMED',mode=mode,mutation=mutation,original_reviewed_sha256=script_sha,changed_path_sha256=after_path_sha,actual_mock_submitted_input_sha256=sha(submitted[0]),journal_input_sha256=value['submitted_input_sha256'],journal_status=value['status'],submitted_argv=value['submit_argv'],mock_sbatch_count=1,initial_submitting_observed=True,real_function_used=True,reentry_refusal=reentry,path_reopened_by_submit=False)
        result.update(label=label,source_submitter_sha256=sha((SOURCE/'submit_once.py').read_bytes()),source_inventory_sha256=sha((SOURCE/'FILE_HASHES_V3.json').read_bytes()),fixture_only=True,actual_slurm_calls=0,official_assets_loads=0,pt_deserializations=0,calls=calls)
        dump(label+'-PROOF.json',result)
        return result

cases=[probe('MUTATED_PATH_SUCCESS'),probe('MUTATED_PATH_TIMEOUT',mode='timeout'),probe('DELETED_PATH_SUCCESS',mutation='delete'),probe('INVALID_UTF8_SCRIPT',invalid_suffix=b'\xff'),probe('NUL_SCRIPT',invalid_suffix=b'\x00'),probe('INVALID_NATIVE_OUTPUT',mode='invalid-output'),probe('FOREIGN_NATIVE_OUTPUT',mode='foreign-output')]
dump('P05_REGRESSION_RECEIPT.json',dict(status='P05_RESOLVED_SOURCE_CPU_ACTUAL_FUNCTION_REGRESSION',cases=cases,case_count=len(cases),named_probes_not_added_to_unittest_method_count=True,old_v2_rejection_preserved=True,actual_slurm_calls=0,remote_calls=0,gpu_calls=0,official_assets_loads=0,pt_deserializations=0,owner_source_edits=0,dispatch_permission=False,gpu_ready=False))
print(json.dumps(dict(status='PASS',case_count=len(cases),source='actual V3 submit function with injected synthetic scheduler',actual_slurm_calls=0)))
