"""Explicit one-auxiliary-GPU wrapper around the existing feature precompute.

Run this separately after CPU preparation. It cannot start actor training or a
retriever; the canonical model/data/cache paths come from the prepared receipt.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from reproduction.a800_common.deploy import digest, write_once
from reproduction.a800_common.deploy_qprobe import validate_intent


def bind_cuda_uuid(gpu_uuid, uuid_reader):
    """Freeze NVML's admitted physical UUID before importing Torch or loading a model."""
    assert gpu_uuid.startswith('GPU-') and gpu_uuid.count('-') == 5
    assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules), 'Torch imported before CUDA UUID binding'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    actual = uuid_reader(1)
    assert actual == [gpu_uuid], 'Actual CUDA UUID differs from the admitted physical feature GPU'
    return actual


def progress(path, value):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.partial')
    write_once(temporary, value)
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ProgressOutput:
    def __init__(self, original, path, identity):
        self.original, self.path, self.identity = original, path, identity
        self.pending = ''

    def write(self, text):
        self.original.write(text);self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('stage') == 'question_feature_precompute':
                progress(self.path, dict(self.identity, **row, observed_unix=time.time(), exit_code=None))
        return len(text)

    def flush(self):
        self.original.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--physical-index', type=int, choices=(5,6), default=6)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args();base = args.base
    assert base.is_absolute() and base.resolve() == base and base.stat().st_uid == os.getuid()
    intent = validate_intent(json.loads((base/'preparation/qprobe-intent.json').read_text()))
    assert str(base) == intent['base'] and args.batch_size > 0
    run = json.loads((base/'logs/run-manifest.json').read_text())
    assert run['completed_updates'] == 0 and run['execution_backend'] == 'ssh_local'
    assert run['source_commit'] == intent['source_commit'] and run['source_sha256'] == intent['source_sha256']
    assert run['execution_host'] == socket.gethostname() and run.get('question_probe') is None
    assert run['initial_model'] == run['reference_model'] == '/data/search-g1/assets/qwen2.5-3b-instruct'
    for path in (base/'data', Path(run['initial_model']), base/'preparation/official-model-manifest-original.json',
                 base/'question-features', base/'cache', base/'tmp'):
        assert str(path).startswith('/data/') and path.is_absolute() and path.resolve() == path, 'Feature assets and outputs must use real remote paths'
    assert os.environ.get('SLURM_JOB_ID') is None
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.physical_index)
    for name in ('pipeline-launch-claim.json','training-launch-claim.json','deployment-launch-receipt.json','process-ledger.json','qprobe-bind-claim.json'):
        assert not (base/'logs'/name).exists(), 'Features precede any pipeline launch or qprobe binding'
    repo = base/'repo'
    assert Path(__file__).resolve() == repo/'reproduction/a800_common/precompute_qprobe.py'
    for name, expected in run['source_sha256'].items():
        assert digest(repo/name) == expected
    text = (Path('/proc')/str(os.getpid())/'stat').read_text();fields = text[text.rfind(')')+2:].split()
    identity = dict(run_id=run['run_id'], source_commit=run['source_commit'], pid=os.getpid(),
                    start_ticks=int(fields[19]), sid=int(fields[3]), uid=os.getuid(),
                    hostname=socket.gethostname(), physical_index=args.physical_index,
                    completed_training_outers=0, target_training_outers=200)
    write_once(base/'logs/qprobe-feature-claim.json', dict(identity, claimed_unix=time.time(), no_automatic_retry=True))
    destination = base/'preparation/question-features-progress.json'
    progress(destination, dict(identity, stage='FEATURE_GPU_ADMISSION_PENDING', completed=0, observed_unix=time.time(), exit_code=None))
    original = sys.stdout
    log = (base/'preparation/question-feature-precompute.log').open('x')
    try:
        gpu_rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        selected = [row for row in csv.reader(io.StringIO(gpu_rows)) if int(row[0].strip()) == args.physical_index]
        assert len(selected) == 1 and int(selected[0][2].strip()) <= 1024 and int(selected[0][3].strip()) == 0, 'Feature GPU is occupied'
        gpu_uuid = selected[0][1].strip()
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True, timeout=15)
        assert not any(row and row[0].strip() == gpu_uuid for row in csv.reader(io.StringIO(apps))), 'Existing compute process occupies feature GPU'
        from reproduction.common8.native_allocation import cuda_uuids
        actual_uuids = bind_cuda_uuid(gpu_uuid, cuda_uuids)
        identity.update(physical_gpu_uuid=gpu_uuid, actual_cuda_uuids=actual_uuids)
        for key in tuple(os.environ):
            if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
                os.environ.pop(key, None)
        os.environ.update(SEARCH_G1_BASE=str(base), SEARCH_G1_TRAINING_WORLD_SIZE='4',
            HF_HOME=str(base/'cache/hf'), XDG_CACHE_HOME=str(base/'cache'), TMPDIR=str(base/'tmp'),
            TRITON_CACHE_DIR=str(base/'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        assert torch.cuda.device_count() == 1, 'Actual feature CUDA topology differs'
        from search_g1.qprobe_prepare import precompute
        sys.stdout = ProgressOutput(log, destination, identity)
        result = precompute(SimpleNamespace(data=str(base/'data'), model=run['initial_model'],
            official_manifest=str(base/'preparation/official-model-manifest-original.json'),
            cache_dir=str(base/'question-features'), device='cuda:0', batch_size=args.batch_size,
            feature_layer=27, max_prompt_tokens=4096, prompt_key='prompt'))
        assert result['count'] == 169028
        result.update(identity, actual_visible_cuda_count=1, physical_gpu_uuid=gpu_uuid, exit_code=0,
                      extra_teacher_warmup_outers=1, checked_unix=time.time())
        write_once(base/'preparation/features-ready.json', result)
        progress(destination, dict(result, completed=169028, observed_unix=time.time()))
        sys.stdout = original
        print(json.dumps(result), flush=True)
    except BaseException as failure:
        progress(destination, dict(identity, stage='FEATURE_PREPARATION_FAILED', failure_type=type(failure).__name__,
                                   observed_unix=time.time(), exit_code=1, no_automatic_retry=True))
        raise
    finally:
        sys.stdout = original
        log.flush();os.fsync(log.fileno());log.close()


if __name__ == '__main__':
    main()
