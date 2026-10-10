"""Exercise native imports, actual CLI composition, CPU Ray and full-index retrieval."""
import importlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

BASE = pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
ROOT = BASE / 'runtime'
OUT = BASE / 'outputs/training-preparation'
c = json.loads((OUT / 'hpc3-config-draft.json').read_text())
sys.path[:0] = [str(ROOT / 'retriever-gpu-lib'), c['candidate'],
               str(ROOT / 'tools/topd-instruct300-launch-v6/runtime'), c['teacher_backend']]

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2))
    print(json.dumps({'phase': name, 'epoch': time.time()}), flush=True)

versions = {}
for name in ['torch', 'transformers', 'vllm', 'flash_attn', 'flash_attn_interface',
             'flash_attn_3_cuda', 'ray', 'tensordict', 'faiss', 'verl.trainer.main_ppo',
             'verl.workers.fsdp_workers', 'verl.workers.rollout.vllm_rollout.vllm_rollout',
             'topd_vllm_teacher.client', 'topd_vllm_teacher.worker', 'topd_vllm_teacher.daemon']:
    m = importlib.import_module(name)
    versions[name] = {'version': getattr(m, '__version__', None), 'path': getattr(m, '__file__', None)}
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.layers.rotary import apply_rotary_emb
import torch
x = torch.arange(24).reshape(2, 3, 4).float()
mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
flat, indices, lengths, maxlen = unpad_input(x, mask)
assert torch.equal(pad_input(flat, indices, 2, 3)[mask.bool()], x[mask.bool()])
save('native-compatibility-validation.json', {'status': 'FA2_FA3_AND_TRAINER_IMPORT_PASS_CPU_PADDING_PASS',
      'modules': versions, 'fa2': importlib.metadata.version('flash-attn'),
      'fa3': importlib.metadata.version('flash-attn-3'), 'gpu_execution_tested': False})
env = dict(os.environ, PYTHONPATH=':'.join(sys.path[:4]))
result = subprocess.run([sys.executable, '-m', 'verl.trainer.main_ppo', '--config-name', 'ppo_baselines', '--cfg', 'job'] + c['training_overrides'],
                        cwd=c['candidate'], env=env, capture_output=True, text=True, timeout=180)
(OUT / 'hydra-composed-config.log').write_text(result.stdout + result.stderr)
if result.returncode: raise RuntimeError('Training CLI configuration failed: ' + result.stderr[-3000:])
save('trainer-cli-validation.json', {'status': 'ACTUAL_TRAINING_CLI_COMPOSITION_PASS_NO_TRAINING', 'returncode': result.returncode})
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
with initialize_config_dir(config_dir=str(pathlib.Path(c['candidate']) / 'verl/trainer/config'), version_base=None):
    resolved = compose(config_name='ppo_baselines', overrides=c['training_overrides'])
(OUT / 'resolved-training-config.yaml').write_text(OmegaConf.to_yaml(resolved))
from compiler_headers import verify
save('compiler-header-validation.json', verify(c['compiler_headers']))
from transformers import AutoConfig, AutoTokenizer
for p in [c['base_model'], c['teacher_path']]:
    AutoConfig.from_pretrained(p, local_files_only=True)
    t = AutoTokenizer.from_pretrained(p, local_files_only=True)
    assert t.encode('Question: What is the capital of France? <search>France</search>')
save('offline-model-loading-validation.json', {'status': 'STUDENT_TEACHER_CONFIG_TOKENIZER_OFFLINE_PASS'})
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl import DataProto
tokenizer = AutoTokenizer.from_pretrained(c['base_model'], local_files_only=True)
dataset = RLHFDataset(c['val_file'], tokenizer, prompt_key=resolved.data.prompt_key,
                     max_prompt_length=resolved.data.max_prompt_length, filter_prompts=True,
                     return_raw_chat=resolved.data.get('return_raw_chat', False), truncation='error',
                     cache_dir=str(BASE / 'environment/cache/verl'))
batch = DataProto.from_single_dict(collate_fn([dataset[0], dataset[1]]))
assert len(batch) == 2 and batch.batch['input_ids'].shape[0] == 2
save('training-dataset-cpu-validation.json', {'status': 'ACTUAL_DATASET_TOKENIZE_COLLATE_DATAPROTO_PASS',
      'validation_rows_after_filter': len(dataset), 'batch_shape': list(batch.batch['input_ids'].shape)})
import ray
from allocator_policy import worker_environment
temp = BASE / 'environment/ray-validation'
temp.mkdir(exist_ok=True)
# Ray Unix socket paths need a short alias; all actual files stay on shared storage.
alias = pathlib.Path('/tmp') / ('topd-ray-' + os.environ['SLURM_JOB_ID'])
alias.symlink_to(temp, target_is_directory=True)
try:
    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False, _temp_dir=str(alias),
             object_store_memory=100 * 1024**2, runtime_env={'env_vars': worker_environment(os.environ)})
    @ray.remote
    def actual_worker(v):
        import torch, flash_attn_interface, flash_attn_3_cuda
        assert os.environ.get('LD_PRELOAD')
        return v + 1
    assert ray.get(actual_worker.remote(41), timeout=90) == 42
    assert ray.get(actual_worker.remote(42), timeout=90) == 43
    save('ray-cpu-validation.json', {'status': 'TWO_ACTUAL_CPU_TASKS_PASS_ZERO_GPU'})
finally:
    ray.shutdown()
    alias.unlink()
import faiss
import numpy as np
from retrieval_server import Encoder
assert hasattr(faiss, 'index_cpu_to_gpu')
encoder = Encoder('e5', str(ROOT / 'assets-root/assets/e5-base-v2'), 'mean', 256, False, 'cpu')
emb = encoder.encode(['What is the capital of France?'])
assert emb.shape == (1, 768) and np.isfinite(emb).all()
save('e5-cpu-validation.json', {'status': 'ORIGINAL_ENCODER_CPU_FORWARD_PASS', 'shape': list(emb.shape)})
index = faiss.read_index(str(ROOT / 'assets-root/assets/wiki/e5_Flat.index'))
assert index.d == 768 and index.ntotal > 1000000
scores, ids = index.search(emb, 3)
assert ids.shape == (1, 3) and (ids >= 0).all() and np.isfinite(scores).all()
selected = set(int(i) for i in ids[0]); docs = {}
with (ROOT / 'assets-root/assets/wiki/wiki-18.jsonl').open() as f:
    for i, line in enumerate(f):
        if i in selected:
            doc = json.loads(line)
            assert isinstance(doc.get('contents'), str)
            docs[str(i)] = {'contents_prefix': doc['contents'][:160]}
            if len(docs) == 3: break
assert len(docs) == 3
save('full-index-cpu-validation.json', {'status': 'FULL_INDEX_READ_SEARCH_CORPUS_MATCH_PASS',
      'dimension': index.d, 'vectors': index.ntotal, 'ids': ids.tolist(), 'scores': scores.tolist(), 'documents': docs})
save('runtime-cpu-complete.json', {'status': 'TRAINING_RUNTIME_CPU_VALIDATION_PASS',
      'gpu_execution_tested': False, 'epoch': time.time()})
