"""Fixed ba9 trained200 final-only handoff; never stops or resumes training."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
TRAIN_SOURCE = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
TRAIN_JOB = 'not-slurm;systemd:' + TRAIN_SOURCE
TRAIN_UNIT = 'sst-tree-nested-branch-fresh200-ba9f6627.service'
RUN = ROOT / 'runs' / ('tree-nested-branch-fresh200-' + TRAIN_JOB)
PACKAGE = ROOT / 'instruct300/candidates' / ('nested-branch-fresh200-' + TRAIN_SOURCE)
PACKAGE_SHA = '514c4473def43e3ce781899d4b62958eee51d752383806420a3ed13a029b0d94'
FROZEN_SOURCE = PACKAGE / 'source/repo'
CONFIG = RUN / 'resolved.yaml'
CONFIG_SHA = '22beaec8aa2b449b496754aacfd9b1e7c89009c9c27a8eff3aef92b768b28b59'
NATIVE = RUN / 'checkpoints/training_state/global_step_200/manifest.json'
HF = RUN / 'checkpoints/actor/global_step_200'
OUTPUT = RUN / 'output/final-eval-fast'
COUNTS = dict(nq=3610, triviaqa=11313, popqa=14267, hotpotqa=7405,
              **{'2wikimultihopqa':12576}, musique=2417, bamboogle=125)
GPU_UUIDS = ('GPU-2c3bdc08-b976-0fb5-79bd-04def458046b', 'GPU-6e00764e-3916-65f2-436e-568aa0c70fc5',
             'GPU-db94211b-42d5-88d0-f626-60e777af3be6', 'GPU-308b13c1-44ea-47b7-7b33-01a928099618',
             'GPU-525a9efc-446e-333d-11ff-e1f2213d15c8', 'GPU-d61d94cd-3d24-fa07-8761-33961a14d995',
             'GPU-f3a967b0-0b4a-4d13-0917-0a4f70aa740b', 'GPU-fca5fcba-4036-0e79-4cb6-ceb9ed121ad5')
REUSED_FILES = {'eval_worker.py':'906095da3e22b4f667fdb36cc635aa52bd8b0d968d116aeb3dcc59f4fda3fcea',
                'eval_merge.py':'9159f0b2021a97cb4aa24bcfe8e67c08c6cafc44f7d567976aae8a851e1b205a',
                'shared_guard.py':'b750ae444454106b002138b79e188769f46562bc95066e9dcfc488acb5740086'}
HANDOFF_STATUS = 'EXPLICIT_NESTED_TRAIN200_FINAL_EVALUATION_HANDOFF_AUTHORIZED'


def require(condition, message):
    if not condition:
        raise ValueError('Nested final handoff: ' + message)


def sha(path):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), 'ordinary file required: '+str(path))
    value = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY|os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda:stream.read(8*1024**2), b''):
            value.update(block)
        after = os.fstat(stream.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
            == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'file changed while held')
    return value.hexdigest()


def read_json(path, expected=None):
    path = Path(path)
    require(sha(path) == expected if expected is not None else True, 'JSON hash changed: '+str(path))
    with os.fdopen(os.open(path, os.O_RDONLY|os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(); after = os.fstat(stream.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
            == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
            and (expected is None or hashlib.sha256(raw).hexdigest() == expected), 'JSON changed while held')
    return json.loads(raw)


def descriptor(item, expected_path=None):
    require(isinstance(item,dict) and set(item)=={'path','sha256'}, 'path/SHA descriptor required')
    path = Path(item['path'])
    require(path.is_absolute() and '..' not in path.parts and path.resolve().is_relative_to(ROOT.resolve()),
            'descriptor must remain in this retained remote task storage')
    require(expected_path is None or path == expected_path, 'descriptor path differs')
    require(re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None, 'full SHA required')
    return read_json(path, item['sha256'])


def verify_code_manifest(tools, manifest, expected):
    tools = Path(tools).resolve(); value = read_json(manifest, expected)
    require(socket.gethostname().split('.')[0]=='h100-10', 'only retained H10018 is permitted')
    commit = value['eval_wrapper_source_commit']
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None and commit != TRAIN_SOURCE
            and tools == ROOT/'instruct300'/('eval-nested-'+commit), 'distinct committed wrapper identity required')
    external={str(ROOT/'tools/tree_shared_owned_runtime_fixed.py'):'4288a4dcfa923df4d8d74f009bf4c56c8446d04fa8ef4202c5e55d3371539ba0',
              str(ROOT/'repo/reproduction/h20/retriever_local.py'):'a78f0b493c538dddc6619e0e462995fd9945c25a187c2165e99d4c10622ce459'}
    require(value['external_files']==external and all(sha(name)==expected for name,expected in external.items()),
            'unchanged own runtime/retriever helpers must remain pinned')
    require(value['schema']=='nested_final_eval_code_v1' and value['training_source_commit']==TRAIN_SOURCE
            and value['operational_changes_independently_reviewed'] is False,
            'wrapper identity or unclaimed operational review differs')
    names = {'contract.py','prepare.py','supervisor.py','dispatch_once.py','build_code_package.py',
             'README.md',*REUSED_FILES}
    require(set(value['files'])==names, 'complete standalone wrapper inventory required')
    for name, expected_sha in value['files'].items():
        require(sha(tools/name)==expected_sha, 'wrapper source changed: '+name)
    require(all(value['files'][name]==digest for name,digest in REUSED_FILES.items()),
            'actually used worker/merge/guard bytes must remain unchanged')
    return value


def training_metadata():
    """JSON-only current-chain completion checks; no native tensor deserialization."""
    require(socket.gethostname().split('.')[0]=='h100-10', 'retained host required')
    require(sha(CONFIG)==CONFIG_SHA, 'actual ba9 configuration changed')
    package = read_json(PACKAGE/'manifest.json',PACKAGE_SHA)
    require(package['code_commit']==TRAIN_SOURCE, 'frozen training package changed')
    for path in (FROZEN_SOURCE,PACKAGE):
        sys.path.insert(0,str(path))
    os.environ.update(SEARCH_R1_BASELINE_COMMIT=TRAIN_SOURCE,SEARCH_R1_BASELINE_CONFIG_SHA256=CONFIG_SHA,
        SEARCH_R1_BASELINE_RUN_ID=RUN.name,SEARCH_R1_BASELINE_OUTPUT=str(RUN/'output'),
        SEARCH_R1_BASELINE_DATA_IDENTITY=package['common_data_identity'],
        SEARCH_R1_MODEL_IDENTITY=package['model_identity']['sha256'],SLURM_JOB_ID=TRAIN_JOB)
    from search_r1.baselines.tree_nested_chain_resume import install_package_environment
    install_package_environment(package)
    import production_contract
    ranks = production_contract.direct_rank_records(RUN,package,CONFIG_SHA,200)
    native, native_sha = production_contract.checkpoint_manifest(RUN,package,CONFIG_SHA,200)
    accounting = native['identity']['guard_continue_accounting']
    require(ranks['completed_outer']==200 and accounting['completed_outer_updates']==200
            and ranks['actual_writes_per_rank']==accounting['total_optimizer_calls_per_rank']
            and ranks['attempt_accounting']['inherited_completed_outers']==1
            and ranks['attempt_accounting']['failed_parent_outer2_in_effective_history'] is False,
            'real effective200 mother1/child2..200 accounting required')
    for record in native['worker_states']:
        rank = str(record['identity']['rank']); calls = accounting['total_optimizer_calls_per_rank'][rank]
        require(record['actual_total_optimizer_calls']==calls and record['observed_adam_steps']==[calls]
                and record['scheduler_last_epoch']==200 and record['full_worker_state_finite'] is True,
                'native200 rank actual optimizer/scheduler metadata differs')
    return package,native,native_sha,ranks


def original_terminal():
    raw = subprocess.check_output(['systemctl','show',TRAIN_UNIT,
        '--property=MainPID,ActiveState,SubState,Result,ExecMainStatus,ControlGroup'],text=True)
    value = dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    require(value.get('MainPID')=='0' and value.get('ActiveState') in ('inactive','failed'),
            'current training must remain running until a separately authorized true200 final-only handoff')
    relative = value.get('ControlGroup','')
    if relative:
        require(relative.startswith('/') and '..' not in Path(relative).parts and TRAIN_UNIT in Path(relative).parts,
                'original own cgroup identity differs')
        scope = Path('/sys/fs/cgroup')/relative.lstrip('/')
        if scope.exists():
            require(not any(p.read_text().strip() for p in scope.rglob('cgroup.procs')), 'original live descendants remain')
    return value


def verify_archive(value):
    require(value['status']=='NESTED_NATIVE_HF200_DURABLE_ARCHIVE_VERIFIED'
            and value['training_source_commit']==TRAIN_SOURCE and value['training_run']==str(RUN)
            and value['actual_completed_outer']==200 and value['native_manifest']['path']==str(NATIVE)
            and value['config_sha256']==CONFIG_SHA and value['native_state_deserialized'] is False,
            'real complete200 durable archive proof required')
    require(sha(NATIVE)==value['native_manifest']['sha256'], 'native200 manifest changed since archive')
    native = read_json(NATIVE)
    ids=native['identity']
    require(native['status']=='COMPLETE_SAME_WORLD_STATE_SAVED'
            and ids['job_id']==TRAIN_JOB and ids['code_commit']==TRAIN_SOURCE and ids['config_sha256']==CONFIG_SHA
            and ids['completed_step']==ids['actual_update_limit']==200
            and value['native_identity']==ids and value['hf_checkpoint']==str(HF)
            and value['hf_identity']==native['hf_identity']
            and value['actual_optimizer_calls_per_rank']==ids['guard_continue_accounting']['total_optimizer_calls_per_rank']
            and value['effective_parent_outer1_included'] is True and value['failed_parent_original_outer2_included'] is False
            and value['all_files_hash_verified'] is True and value['all_files_fsynced'] is True,
            'archive source/effective200/optimizer/HF/durability binding differs')
    paths = {str(NATIVE),native['driver']['path'],*[r['path'] for r in native['worker_states']],
             *[str(HF/name) for name in native['hf_identity']['files']]}
    require(set(value['files'])==paths, 'archive omitted a native/HF file or added unrelated state')
    for path,record in value['files'].items():
        require(Path(path).stat().st_size==record['bytes'] and sha(path)==record['sha256'], 'archived checkpoint changed: '+path)
    return native


def verify_handoff(path, expected):
    return verify_handoff_value(read_json(path,expected),path)


def verify_handoff_value(value, path):
    require(Path(path).is_absolute() and Path(path).resolve().parent==OUTPUT.resolve(), 'handoff must stay in own final-eval directory')
    require(value['status']==HANDOFF_STATUS and value['training_source_commit']==TRAIN_SOURCE
            and value['training_run']==str(RUN) and value['training_unit']==TRAIN_UNIT
            and value['actual_training_completed_outer']==200 and value['exact_executor_resume'] is False,
            'explicit fixed-chain trained200 handoff required')
    authority = value['authority']
    require(authority['scope']=='only_final_evaluation_after_completed_training200'
            and isinstance(authority['human_instruction'],str) and authority['human_instruction'].strip()
            and isinstance(authority['original_instruction_time'],str) and authority['original_instruction_time'].strip()
            and value['training_stop_or_resume_by_wrapper'] is False,
            'original explicit human final-only handoff authority required; wrapper cannot stop training')
    archive = descriptor(value['archive'],OUTPUT/'archive-proof.json')
    native = verify_archive(archive)
    assets = descriptor(value['asset_manifest'],OUTPUT/'asset-manifest.json')
    prefix = descriptor(value['prefix'],OUTPUT/'bound-prefix-empty.json')
    from eval_merge import identities
    require({r['name']:r['sha256'] for r in assets['model_files']}==native['hf_identity']['files']
            and assets['config_sha256']==CONFIG_SHA and assets['dataset_count']==51713
            and assets['expected_source_counts']==COUNTS and prefix['rows']==[]
            and all(prefix[key]==val for key,val in identities(assets).items())
            and value['score_output']==str(OUTPUT/'scores'), 'new HF200/data/empty prefix binding differs')
    done = read_json(str(OUTPUT/'bound-prefix-empty.json')+'.done.json')
    require(done['count']==0 and done['sha256']==value['prefix']['sha256']
            and done['actual_trained_outer']==200 and done['identity']==identities(assets), 'empty prefix durable proof differs')
    terminal = descriptor(value['original_terminal'],OUTPUT/'original-terminal.json')
    live = original_terminal()
    require(terminal['unit']==TRAIN_UNIT and terminal['systemd']==live
            and terminal['normal_exit_claimed'] is False and terminal['original_final_evaluation_complete'] is False,
            'truthful original interrupted final-only terminal required')
    original = descriptor(value['original_run_result'],RUN/'run-result.json')
    require(original['status']=='FAILED' and original['code_commit']==TRAIN_SOURCE
            and original['resumed_current_chain'] is True and original['own_cgroup_live_residual_empty'] is True
            and original['actual_progress_evidence']['actual_completed_outer_from_attempt_results']==200,
            'original supervisor terminal must retain real trained200 and actual cleanup; no synthesized exit0')
    return value
