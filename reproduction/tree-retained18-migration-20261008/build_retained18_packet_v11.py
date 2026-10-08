"""Materialize a code-only retained18 continuation packet, without running it."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
OLD=WORKSPACE/'work/tree-fresh200-direct-formal-a18f420/package-code-only-v9'
OUT=HERE/'package-code-only-v11'
SOURCE='6efb60466a233f70bf8f50de26df32012f49b4bc'
ROOT='/dataset/shaoshitong/search-r1-rl/tree-grpo'
REMOTE_PACKAGE=ROOT+'/instruct300/candidates/current200-retained18-v11-'+SOURCE[:7]
AUTH=ROOT+'/instruct300/CURRENT200_RETAINED18_HUMAN_AUTHORITY-V11-'+SOURCE[:7]+'.json'
RELEASE=ROOT+'/instruct300/CURRENT200_RETAINED18_DIRECT_RELEASE-V11-'+SOURCE[:7]+'.json'
INTENT=ROOT+'/instruct300/SUBMISSION_ONCE-retained18-current200-v11-'+SOURCE[:7]+'.json'
UUIDS=['GPU-2c3bdc08-b976-0fb5-79bd-04def458046b','GPU-6e00764e-3916-65f2-436e-568aa0c70fc5','GPU-db94211b-42d5-88d0-f626-60e777af3be6','GPU-308b13c1-44ea-47b7-7b33-01a928099618','GPU-525a9efc-446e-333d-11ff-e1f2213d15c8','GPU-d61d94cd-3d24-fa07-8761-33961a14d995','GPU-f3a967b0-0b4a-4d13-0917-0a4f70aa740b','GPU-fca5fcba-4036-0e79-4cb6-ceb9ed121ad5']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    if OUT.exists():raise RuntimeError('Existing packet: preserve rather than overwrite')
    shutil.copytree(OLD,OUT,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    shutil.rmtree(OUT/'source/repo')
    (OUT/'source/repo').mkdir()
    raw=subprocess.check_output(['git','archive',SOURCE],cwd=HERE/'source')
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for item in archive.getmembers():
            if Path(item.name).is_absolute() or '..' in Path(item.name).parts or item.issym() or item.islnk():
                raise RuntimeError('Unexpected archive path/link')
        archive.extractall(OUT/'source/repo')
    # Active wrappers only; historical receipts and saved parent identities stay intact.
    for p in OUT.iterdir():
        if p.suffix not in ('.py','.sh','.sbatch'):continue
        text=p.read_text().replace('h100-9','h100-10').replace('same_h100_9','same_h100_10')
        text=text.replace("['packet_revision']==9","['packet_revision']==11").replace("get('packet_revision')==9","get('packet_revision')==11")
        text=text.replace("SOURCE = 'e7236b7d8729482aad864419fe986066643f3be5'",f"SOURCE = '{SOURCE}'")
        text=text.replace("package.resolve().is_relative_to(ROOT/'instruct300/candidates')","package.resolve().is_relative_to((ROOT/'instruct300/candidates').resolve())")
        text=text.replace("package.is_relative_to(ROOT/'instruct300/candidates')","package.is_relative_to((ROOT/'instruct300/candidates').resolve())") if p.name=='launch.sbatch' else text
        p.write_text(text)
    # Both sides must compare the same physical output directory after migration.
    observer=OUT/'validation_cost_hooks_instruct300.py'
    old="path==pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs')/('tree-current200-continue-'+job)/'output'"
    new="path==(pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs')/('tree-current200-continue-'+job)/'output').resolve()"
    text=observer.read_text()
    if text.count(old)!=1:raise RuntimeError('Unexpected exact observer path assertion')
    observer.write_text(text.replace(old,new,1))
    # Isolated node guard: no edits to historical tools or another task.
    text=(WORKSPACE/'work/resource_guard_identity.py').read_text().replace("=='h100-9'","=='h100-10'")
    text=text.replace("pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/expected-gpu-uuids.json')","(pathlib.Path(__file__).parent/'expected-gpu-uuids-retained18.json')")
    (OUT/'resource_guard_retained18.py').write_text(text)
    write(OUT/'expected-gpu-uuids-retained18.json',UUIDS)
    p=OUT/'supervisor.py';text=p.read_text()
    text=text.replace("sha(ROOT/'tools/resource_guard_identity.py')","sha(TOOLS/'resource_guard_retained18.py')")
    text=text.replace("str(ROOT/'tools/resource_guard_identity.py')","str(TOOLS/'resource_guard_retained18.py')")
    text=text.replace("PACKAGE['external_files'][str(TOOLS/'resource_guard_retained18.py')]","PACKAGE['files']['resource_guard_retained18.py']")
    p.write_text(text)
    # Original9 PIDs are historical identities, never reinterpreted as node18 PIDs.
    p=OUT/'calibration_contract.py';text=p.read_text().replace('def protected_processes():','def historical_original9_process_observer():',1)
    insert='''def protected_processes():
    """Original9 is retired; its PID namespace is not the retained18 namespace."""
    rows=[dict(pid=pid,expected_starttime_jiffies=start,status='RETIRED_ORIGINAL_NODE_IDENTITY_UNOBSERVABLE',
               actual_starttime_jiffies=None,original_alive=False,identity_known=False,original_node='h100-9',
               current_node='h100-10',process_signals_sent=False) for pid,start in PROTECTED.items()]
    return dict(status='ORIGINAL_NODE_RETIRED_CURRENT_FOREIGN_GUARD_REMAINS',originals=rows,
        original_eight_pid_start_times_preserved=None,missing_original_pids=list(PROTECTED),
        current_foreign_processes_untouched=True,process_signals_sent=False)


'''
    text=text.replace('def final_gpu_process_observation(',insert+'def final_gpu_process_observation(',1);p.write_text(text)
    shutil.copyfile(HERE/'submit_retained18_v11.py',OUT/'submit_once.py')
    p=OUT/'render_dispatch.py';text=p.read_text();text=text.replace("'#SBATCH --exclusive',","'#SBATCH --exclusive',\n        '#SBATCH --output='+str(ROOT/'instruct300/node-move-logs/%j.out'),\n        '#SBATCH --error='+str(ROOT/'instruct300/node-move-logs/%j.err'),")
    p.write_text(text)
    v=json.loads((OLD/'manifest.json').read_text());oldsha=sha(OLD/'manifest.json')
    v['code_commit']=SOURCE;v['packet_revision']=11;v['materialized_at_utc']=datetime.now(timezone.utc).isoformat()
    v['dispatch']['nodes']=['h100-10'];v['dispatch']['once_intent']=INTENT
    v['configuration_overrides']=[x.replace('baselines.local_runtime_host=h100-9','baselines.local_runtime_host=h100-10') for x in v['configuration_overrides']]
    v['external_files'].pop(ROOT+'/tools/resource_guard_identity.py')
    v['supersedes_failed_packet_manifest_sha256']='13268301b6718d77e1159a60526c67d9e139f6d05310438ac3f5b2b10f642cf9'
    failed=json.loads((HERE/'failed-predecessor3334.json').read_text())
    v['failed_predecessor3334']=dict(path=failed['path'],sha256=failed['sha256'])
    v['path_fix_scope']='Only resolve both sides of the exact observer output-directory equality; retain fixed run binding'
    v['core_source_bytes_changed']=False
    v['retired_original3333_packet_manifest_sha256']=oldsha
    v['retained18_migration']=dict(original_node='h100-9',original_ip='10.33.0.13',retirement_date='2026-10-08',
        destination_node='h100-10',destination_ip='10.33.0.18',destination_retained=True,
        preserved_parent_completed_outer=4,unpreserved3333_progress_unknown=True,original3333_terminal_state=None,
        original3333_cleanup_verified=False,same_cluster_assets=True,assets_local_on_destination=True,
        output_server='10.33.0.18',output_server_retained=True,storage_receipt_sha256=sha(HERE/'output-storage-verified.json'),
        asset_copy_required_status='LOCAL_ASSETS_AND_PARENT_FILES_COPIED_VERIFIED',exact_executor_resume=False,
        foreign_job3304_must_be_scheduled_around=False,foreign_job3304_cancelled_by_latest_human=True,extra_short_training_added=False)
    for key in ('predecessor_terminal_cleanup_proof','terminal_proof_closure'):
        v['historical_'+key]=v.pop(key)
    v['historical_source_selection']=v['source_selection']
    v['source_selection']=dict(protocol='git_archive_retained18_host_only_patch',code_commit=SOURCE,
        selected_files={str(p.relative_to(OUT/'source/repo')):sha(p) for p in sorted((OUT/'source/repo').rglob('*')) if p.is_file()},
        contains_model_data_index_checkpoint_assets=False)
    v['source_selection']['selected_count']=len(v['source_selection']['selected_files'])
    v['new_cpu_tests_executed']=False;v['new_tests_executed']=False;v['new_reviews_executed']=False
    v['core_source_bytes_changed']=False
    v['background_lifecycle']=dict(original_node='h100-9',original_eight_identity_preservation=None,
        reason='Retired node PID namespace unavailable; no node18 PID identity substitution',
        current_node_foreign_MEM_SM_gate_percent=15,foreign_processes_signaled=False)
    v['files']={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    v['wrapper_files']={k:h for k,h in v['files'].items() if not k.startswith('source/')}
    authority=dict(status=v['execution_mode'],scope=v['scope'],packet_revision=11,code_commit=SOURCE,
        current_chain_resume=v['current_chain_resume'],same_instance_continuation=True,
        completed_outer_goal=200,final_evaluation_rows=51713,production_thresholds_accepted=False,
        independent_acceptance=False,training_accepted=False,original_sampler_actor_gate_accepted=False,
        configured_prospective_thresholds=v['configured_prospective_thresholds'],
        human_instruction='2026-10-08用户明确：原节点属于退役节点则换到保留10.33.0.18；同H100共享盘可用。继续有效fresh200目标，从同一链实际保存的rolling4继续，无新增0-5阶段。最新人工明确取消OPSD3304并继续Tree；3334因迁移路径比较错误在更新前失败，保留实际失败凭证后修复并从同一rolling4继续。',
        human_instruction_original_time='2026-10-08 Asia/Shanghai; exact message timestamp unavailable',
        retained18_migration=v['retained18_migration'])
    authfile=HERE/'human-authority-v11.json';write(authfile,authority)
    v['human_training_authority']=dict(path=AUTH,sha256=sha(authfile));write(OUT/'manifest.json',v)
    release=dict(status=v['execution_mode'],execution_mode=v['execution_mode'],scope=v['scope'],packet_revision=11,
        code_commit=SOURCE,current_chain_resume=v['current_chain_resume'],manifest_sha256=sha(OUT/'manifest.json'),
        human_training_authority_sha256=sha(authfile),cluster_name='wuhan-h100',node_list='h100-10',
        dispatch_renderer_sha256=v['files']['render_dispatch.py'],dispatch_submitter_sha256=v['files']['submit_once.py'],
        dispatch_intent_path=INTENT,production_thresholds_accepted=False,independent_acceptance=False,
        training_accepted=False,original_sampler_actor_gate_accepted=False,retained18_migration=v['retained18_migration'])
    releasefile=HERE/'direct-release-v11.json';write(releasefile,release)
    spec=dict(package=REMOTE_PACKAGE,manifest_sha=sha(OUT/'manifest.json'),release_path=RELEASE,
        release_sha=sha(releasefile),authority_path=AUTH,authority_sha=sha(authfile),once_intent=INTENT,
        source_commit=SOURCE,local_package=str(OUT),remote_submitted=False)
    write(HERE/'deployment.json',spec)
    print(json.dumps(spec))

if __name__=='__main__':main()
