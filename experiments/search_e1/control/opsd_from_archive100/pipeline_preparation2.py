"""Owner-started preparation-2 pipeline, guarded against live-job overlap.

This controller never stops another process. The owner must first preserve and
stop only preparation-1 and provide the reviewed stop receipt. A shared lock
plus real /proc and GPU process checks prevent overlapping or resumed GRPO.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from prepare_preparation2 import (DEFAULT_BASE, DEFAULT_CANDIDATE, CANDIDATE_PID,
    GPU7_UUID, assert_opsd_unstarted, read_json, sha, verify_candidate,
    verify_manifest, write_new_json)


def require_stop_and_owned_gpu_release(base):
    proof = read_json(base / 'receipts/preparation-1-stopped-for-preparation-2.json')
    old = read_json(base / 'receipts/new-run.json')
    if proof.get('stopped') is not True or proof.get('opsd_updates') != 0:
        raise RuntimeError('Root preparation-1 stop/zero-update proof is required')
    if proof.get('controller_pid') != old['pid'] or proof.get('run_id') != old['run_id']:
        raise RuntimeError('Stop proof belongs to a different controller/run')
    if proof.get('scope') != 'preparation-1 only; OPSD updates 0':
        raise RuntimeError('Stop scope must cover only preparation-1, not another experiment')
    preserved_path = base/'receipts/preparation-1-final-pipeline-stage.json'
    preserved = read_json(preserved_path)
    if preserved.get('controller_pid') != old['pid']:
        raise RuntimeError('Preserved final preparation-1 stage belongs to a different controller')
    if proof.get('final_pipeline_stage_sha256') != sha(preserved_path):
        raise RuntimeError('Root stop proof must bind the preserved final preparation-1 stage SHA')
    owned = set(proof.get('owned_pids', []))
    if old['pid'] not in owned or not owned or CANDIDATE_PID in owned:
        raise RuntimeError('Invalid preparation-1 ownership set')
    for pid in owned:
        process = Path('/proc') / str(pid)
        if process.exists():
            # An unreaped zombie is terminal, but GPU checks below remain
            # authoritative. Any live old process blocks this new controller.
            stat = (process / 'stat').read_text().rsplit(')',1)[1].split()
            if stat[0] != 'Z':
                raise RuntimeError('An old owned preparation process remains live: ' + str(pid))
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    apps = [[piece.strip() for piece in line.split(',')] for line in lines if line.strip()]
    if not apps or any(len(row) != 3 or row[0] != GPU7_UUID or row[1] != str(CANDIDATE_PID) for row in apps):
        raise RuntimeError('GPUs are not released exclusively to the approved GPU7 candidate; no foreign process will be stopped')
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    rows = [[piece.strip() for piece in line.split(',')] for line in cards if line.strip()]
    if sorted(int(row[0]) for row in rows) != list(range(8)) or not any(row[0]=='7' and row[1]==GPU7_UUID for row in rows):
        raise RuntimeError('Allocated eight-card physical GPU inventory changed')
    if any(int(row[2]) < 1024 for row in rows):
        raise RuntimeError('Actual memory.free below 1 GiB before pipeline initialization')
    return dict(stop_proof=proof, actual_compute_apps=apps, actual_cards=rows, old_processes_terminal=True)


def main():
    base = DEFAULT_BASE
    # Same inode/lock as preparation-1; no automatic termination or wait/restart.
    lock = (base / 'experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (base/'receipts/preparation-1-final-pipeline-stage.json').exists():
        raise RuntimeError('Root must preserve final preparation-1 stage before replacing the stable status pointer')
    stage_path = base / 'receipts/preparation-2-pipeline-stage.json'
    stable_stage_path = base / 'receipts/pipeline-stage.json'
    run_id = uuid.uuid4().hex
    def phase(value, **extra):
        data = dict(phase=value, time=time.time(), controller_pid=os.getpid(), run_id=run_id,
            generation=2, status_scope='Latest active pipeline pointer; preparation-1 final stage preserved separately', **extra)
        encoded = json.dumps(data, indent=2)
        for path in [stage_path,stable_stage_path]:
            temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
            temporary.write_text(encoded); temporary.replace(path)
        print('PHASE',value,flush=True)
    try:
        phase('PREPARATION2_VERIFYING_OWNER_STOP_AND_ZERO_OPSD')
        config_path = Path(os.environ.get('E1_PREPARATION_CONFIG', str(base / 'receipts/preparation-2-config.json')))
        if config_path not in [base/'receipts/preparation-2-config.json', base/'receipts/preparation-2-config-v2.json']:
            raise RuntimeError('Unreviewed preparation configuration path')
        config = read_json(config_path)
        if config.get('prepared') is not True or config.get('source_model') != str(base / 'assets/archive100'):
            raise RuntimeError('Prepared archive100 configuration is missing or changed')
        required_paths = dict(rollout_output=str(base/'runs/preparation-2-rollouts.jsonl'),
            pair_output=str(base/'runs/preparation-2-pairs.jsonl'), opsd_output=str(base/'runs/opsd'),
            final_output=str(base/'runs/preparation-2-final-test.jsonl'))
        if any(config.get(key) != value for key,value in required_paths.items()):
            raise RuntimeError('Prepared output paths exceed the reviewed migration scope')
        if config.get('retrievers') != dict(preparation='http://10.120.17.101:18004/retrieve',
            opsd_baseline_validation='http://10.120.17.101:18004/retrieve',
            final_eight_gpu_vllm='http://10.120.17.101:18004/retrieve'):
            raise RuntimeError('Prepared endpoint selection changed without a new reviewed controller')
        execution = config.get('execution_change',{})
        if execution.get('num_gpu_blocks_override') != 22671 or execution.get('inference_utilization') != .85 or execution.get('cpu_blocks_expected_unchanged') != 7281 or execution.get('communication_timeout_hours') != 24 or execution.get('eager') is not True:
            raise RuntimeError('Prepared physical execution settings changed')
        for path, expected in config['controls'].items():
            if sha(Path(path)) != expected:
                raise RuntimeError('Prepared control source changed: ' + path)
        assert_opsd_unstarted(base)
        stop = require_stop_and_owned_gpu_release(base)
        gate = verify_candidate(DEFAULT_CANDIDATE, require_ready=True)
        runtime = Path(config['runtime'])
        if runtime != base / 'runtime/opsd-preparation2':
            raise RuntimeError('Unreviewed preparation runtime')
        manifest_path = base / 'receipts/preparation-2-runtime.sha256.json'
        if sha(manifest_path) != config['new_manifest_sha256']:
            raise RuntimeError('Prepared runtime manifest changed')
        verify_manifest(runtime, read_json(manifest_path))
        if sha(base / 'receipts/opsd-runtime.sha256.json') != config['source_manifest_sha256']:
            raise RuntimeError('Original frozen manifest changed')
        verify_manifest(base / 'runtime/opsd', read_json(base / 'receipts/opsd-runtime.sha256.json'))
        for dataset in config['datasets'].values():
            if sha(Path(dataset['path'])) != dataset['sha256']:
                raise RuntimeError('Original dataset content changed: ' + dataset['path'])
        model_receipt = base / 'receipts/archive100-transfer-accepted.json'
        if sha(model_receipt) != config['source_model_transfer_receipt_sha256']:
            raise RuntimeError('Archive100 source identity receipt changed')
        transfer = read_json(model_receipt)
        if transfer.get('accepted') is not True or transfer.get('target') != config['source_model']:
            raise RuntimeError('Archive100 accepted transfer is missing')
        for filename, evidence in transfer['files'].items():
            path = Path(config['source_model']) / filename
            if path.stat().st_size != evidence['bytes'] or sha(path) != evidence['sha256']:
                raise RuntimeError('Archive100 source weight/tokenizer changed: ' + filename)
        for key in ['rollout_output','pair_output','opsd_output','final_output']:
            if Path(config[key]).exists():
                raise RuntimeError('Output already exists; preserve and inspect instead of restarting: ' + config[key])
        env = dict(os.environ,E1_BASE=str(base),E1_PROFILE='opsd100-preparation2',
            CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
            PYTHONPATH=str(runtime/'upstream/Search-R1')+':'+str(runtime),
            HF_HOME=str(base/'hf-cache'),HF_DATASETS_CACHE=str(base/'hf-datasets'),
            XDG_CACHE_HOME=str(base/'cache'),TMPDIR=str(base/'tmp'),
            HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONUNBUFFERED='1',
            TOKENIZERS_PARALLELISM='false',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',
            E1_INFERENCE_UTIL='.85',WANDB_MODE='offline',E1_RUN_ID=run_id)
        # Do not restore GRPO optimizer/scheduler or reuse preparation-1 outputs.
        run_receipt = dict(run_id=run_id, controller_pid=os.getpid(), time=time.time(),
            controller_pgid=os.getpgrp(), process_uid=os.getuid(),
            config_path=str(config_path), config_sha256=sha(config_path), source_grpo_step=100,
            actual_opsd_updates_at_start=0, additional_grpo=False, reset_grpo_optimizer=True,
            owner_stop_and_gpu_release=stop, candidate=gate, runtime=str(runtime),
            source_model=config['source_model'], scope='New full preparation; first scientific OPSD run remains runs/opsd')
        write_new_json(base/'receipts/preparation-2-run.json',run_receipt)
        py=str(base/'env/bin/python')
        def run(arguments, log_name):
            with (base/'logs'/log_name).open('x') as log:
                child = subprocess.Popen(arguments,env=env,stdout=log,stderr=subprocess.STDOUT,
                                         start_new_session=True)
                ownership = dict(time=time.time(), run_id=run_id, controller_pid=os.getpid(),
                    uid=os.getuid(), child_pid=child.pid, child_pgid=child.pid,
                    argv=arguments, log_path=str(base/'logs'/log_name),
                    scope='Only this new preparation-2 stage and E1_RUN_ID descendants; never another job')
                path = base/'receipts/preparation-2-active-child.json'
                temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
                temporary.write_text(json.dumps(ownership,indent=2)); temporary.replace(path)
                with (base/'receipts/preparation-2-child-history.jsonl').open('a') as history:
                    history.write(json.dumps(ownership)+'\n')
                code = child.wait()
                with (base/'receipts/preparation-2-child-history.jsonl').open('a') as history:
                    history.write(json.dumps(dict(ownership,time=time.time(),exit_code=code,event='exit'))+'\n')
                if code != 0:
                    raise subprocess.CalledProcessError(code,arguments)
        phase('OPSD100_PREPARATION2_FULL_FRESH_TRAJECTORIES')
        run([py,'-m','search_e1.rollout','--model',config['source_model'],'--data',str(base/'data/train.jsonl'),
            '--output',config['rollout_output'],'--k','5','--backend','vllm','--workers','8',
            '--retriever-url',config['retrievers']['preparation']],'preparation-2-rollouts.log')
        phase('OPSD100_PREPARATION2_MINE_FULL_PAIRS')
        run([py,'-m','search_e1.mine','--rollouts',config['rollout_output'],'--output',config['pair_output']],
            'preparation-2-mine.log')
        # Native global validation keeps validation[rank::world]: 64 questions
        # on each of eight ranks, reduced over the original full 512. Exact
        # per-request retrieval operations stay unchanged through admitted
        # 18004; this avoids re-transferring the index eight times for the
        # eight concurrent ranks. Keep the public 18001 relay untouched.
        phase('OPSD100_PREPARATION2_TRAIN_AND_BASELINE_VALIDATION')
        verify_candidate(DEFAULT_CANDIDATE, require_ready=True)
        run([py,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','search_e1.opsd',
            '--model',config['source_model'],'--pairs',config['pair_output'],
            '--validation',str(base/'data/validation.jsonl'),'--accumulation','4',
            '--output',config['opsd_output'],'--retriever-url',config['retrievers']['opsd_baseline_validation']],
            'preparation-2-opsd.log')
        phase('OPSD100_PREPARATION2_FINAL_SEVEN_BENCHMARK_EVALUATION')
        verify_candidate(DEFAULT_CANDIDATE, require_ready=True)
        run([py,'-m','search_e1.rollout','--model',str(base/'runs/opsd/merged'),
            '--data',str(base/'data/test.jsonl'),'--output',config['final_output'],'--k','1',
            '--backend','vllm','--workers','8','--temperature','0',
            '--retriever-url',config['retrievers']['final_eight_gpu_vllm']],'preparation-2-final-test.log')
        phase('COMPLETED_OPSD_AND_PREPARATION2_EVALUATION')
    except Exception as error:
        phase('FAILED_PREPARATION2', error_type=type(error).__name__, reason=str(error))
        raise


if __name__ == '__main__':
    main()
