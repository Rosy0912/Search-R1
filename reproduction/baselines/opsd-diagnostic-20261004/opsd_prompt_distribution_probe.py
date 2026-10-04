"""Read-only, single-owned-H100 initial-state distribution diagnostic.

No retrieval, generation, optimizer, training, or search-effect acceptance.
Models are resident sequentially. All model forwards are inference-only;
CPU gradients below concern saved logit vectors, never model parameters.
"""
import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
BASE = ROOT / 'assets-root/assets/qwen2.5-3b'
HF5 = ROOT / 'runs/opsd-formal-3007/output/checkpoints/opsd-gpu-formal500/actor/global_step_5'
TRAIN = ROOT / 'runs/opsd-formal-3007/output/common-data/e70908e713d6c3a85b06/train.parquet'
TRAINING_COMMIT = '8e60976d364ab7d7ae51714415813c1359dafe9a'
TAGS = ('<think>', '<search>', '<answer>')
BASE_PINS_SHA = '105c9616c25c8b574608fc0a05d467ce05efd7110ffdbddf945a816f42e52d5c'
CASE_NAMES = ('student_base', 'student_hf5', 'teacher_legacy', 'teacher_user_context')
HF5_SHARDS = {
 'model-00001-of-00003.safetensors': 'e26d87c9270778dc7452f88061a3d805228d494c33386876e9188a1d9e6d7282',
 'model-00002-of-00003.safetensors': '7fccc1a617479c2ceccd73b828060b41da088f081c5097b53bc295037029406b',
 'model-00003-of-00003.safetensors': 'b1132ad73397a0eb6c7d1e11bcf37bd0065734eb0c648fb07dc2bae11396fe13',
}
ORIGINAL_BASE_SHARDS = {
 'model-00001-of-00002.safetensors': 'f9558df91d3b89b4826e4db37439edb52f1d62a4fd602685013e7ca6b9f60f8f',
 'model-00002-of-00002.safetensors': '51410930d5cf19a998fdb17ef0c46e4d9ace72c97a975a3331395a8a500f5edb',
}
BASE_FILE_NAMES = {'config.json', 'generation_config.json', 'merges.txt', 'model.safetensors.index.json',
 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', *ORIGINAL_BASE_SHARDS}


def digest(path):
 h = hashlib.sha256()
 with Path(path).open('rb') as handle:
  for block in iter(lambda: handle.read(8 * 1024 * 1024), b''): h.update(block)
 return h.hexdigest()


def plain(value):
 import numpy as np
 if isinstance(value, np.ndarray): return plain(value.tolist())
 if isinstance(value, np.generic): return plain(value.item())
 if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
 if isinstance(value, (list, tuple)): return [plain(v) for v in value]
 if value is None or isinstance(value, (str, int, bool)): return value
 if isinstance(value, float) and math.isfinite(value): return value
 raise ValueError('Nonfinite or unsupported evidence')


def encoded(value):
 return json.dumps(plain(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def loads(raw):
 def unique(items):
  result = {}
  for key, value in items:
   if key in result: raise ValueError('Duplicate JSON key')
   result[key] = value
  return result
 return plain(json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON'))))


def write_new(path, value):
 with Path(path).open('xb') as handle: handle.write(encoded(value) + b'\n')


def bound(record, scope=None):
 if set(record) != {'path', 'sha256'} or not re.fullmatch('[0-9a-f]{64}', record['sha256']): raise ValueError('Exact external path/SHA required')
 path = Path(record['path'])
 if not path.is_absolute() or scope is not None and not path.resolve().is_relative_to(Path(scope).resolve()): raise ValueError('Evidence path escapes prescribed scope')
 if digest(path) != record['sha256']: raise ValueError('External content pin differs')
 return path.resolve()


def verify_binding(binding):
 fields = {'version', 'scope', 'training_commit', 'questions', 'selection', 'train', 'source', 'probe_sha256', 'models', 'base_pins'}
 if set(binding) != fields or binding['version'] != 1 or binding['scope'] != 'train-initial-state-distribution-only': raise ValueError('Wrong diagnostic binding schema')
 if binding['training_commit'] != TRAINING_COMMIT or binding['questions'] != 64 or binding['selection'] != 'first64-ordered-train-rows': raise ValueError('Only fixed 64 training initial states allowed')
 if Path(binding['train']['path']) != TRAIN: raise ValueError('Validation/test or different training split prohibited')
 if binding['probe_sha256'] != digest(Path(__file__)): raise ValueError('Actual executing probe differs from external pin')
 bound(binding['train'], ROOT); bound(binding['source'], ROOT)
 if binding['base_pins']['sha256'] != BASE_PINS_SHA: raise ValueError('Original accepted nine-file base pins required')
 original = loads(bound(binding['base_pins'], ROOT).read_bytes())
 if original['directory'] != str(BASE) or binding['models']['base']['files'] != original['files']: raise ValueError('Actual base differs from original accepted assets')
 if set(binding['models']) != {'base', 'hf5'}: raise ValueError('Only original base and actual step5 allowed')
 indices = {}
 for role, wanted in [('base', BASE), ('hf5', HF5)]:
  record = binding['models'][role]
  if set(record) != {'directory', 'files'} or Path(record['directory']) != wanted: raise ValueError('Model role/path substituted (no HF80/HF500)')
  names = {p.name for p in wanted.iterdir() if p.is_file()}
  if names != set(record['files']) or not {'config.json', 'tokenizer.json', 'model.safetensors.index.json'} <= names: raise ValueError('Incomplete actual model file inventory')
  if role == 'base' and names != BASE_FILE_NAMES: raise ValueError('Original base nine-file inventory required')
  origin = ORIGINAL_BASE_SHARDS if role == 'base' else HF5_SHARDS
  if any(record['files'].get(name) != sha for name, sha in origin.items()): raise ValueError('Original base/HF5 accepted shard identity changed')
  for name, sha in record['files'].items():
   if Path(name).name != name: raise ValueError('Model file escapes role')
   bound({'path': str(wanted / name), 'sha256': sha}, wanted)
  index = loads((wanted / 'model.safetensors.index.json').read_bytes())['weight_map']
  if role == 'hf5' and record['files']['model.safetensors.index.json'] != '3a899e7f6ef595e15c34b5ffc7e4a1df6131d7c371418b1a6e8f823ab8a7302d': raise ValueError('Original HF5 index SHA differs')
  validate_role_index(role,index,origin)
  indices[role] = index
  if loads((wanted/'config.json').read_bytes())['tie_word_embeddings'] is not True: raise ValueError('Actual fixed model embeddings must be tied')
 if set(indices['hf5']) != set(indices['base']) | {'lm_head.weight'}: raise ValueError('HF5/base exact state-dict key relation differs')
 return binding


def validate_role_index(role, index, shards):
 expected = {'model.embed_tokens.weight','model.norm.weight'}
 per_layer = ('self_attn.q_proj.weight','self_attn.q_proj.bias','self_attn.k_proj.weight','self_attn.k_proj.bias','self_attn.v_proj.weight','self_attn.v_proj.bias','self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','input_layernorm.weight','post_attention_layernorm.weight')
 expected |= {f'model.layers.{i}.{name}' for i in range(36) for name in per_layer}
 if role == 'hf5': expected.add('lm_head.weight')
 elif role != 'base': raise ValueError('Unknown model role')
 if set(index) != expected or set(index.values()) != set(shards): raise ValueError('Actual role-specific434/435 fixed model index differs')


def extract_functions(source):
 """Compile only the externally pinned functions; no arbitrary repo imports."""
 import torch
 import torch.nn.functional as F
 tree = ast.parse(Path(source).read_text())
 functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
 if not {'teacher_inputs', 'generalized_jsd'} <= set(functions): raise ValueError('Required actual source functions missing')
 def padded(rows, pad):
  width = max(1, max(map(len, rows), default=0))
  result = torch.full((len(rows), width), pad, dtype=torch.long)
  for i, row in enumerate(rows):
   if row: result[i, :len(row)] = torch.tensor(row)
  return result
 selected = [functions['teacher_inputs'], functions['generalized_jsd']]
 # teacher_inputs has exactly one reviewed relative dependency: padded.
 imports = [n for n in selected[0].body if isinstance(n, ast.ImportFrom)]
 if len(imports) != 1 or imports[0].level != 1 or imports[0].module != 'rollout' or [n.name for n in imports[0].names] != ['padded']: raise ValueError('Unknown teacher helper dependency')
 selected[0].body = [n for n in selected[0].body if n is not imports[0]]
 if any(isinstance(n, (ast.Import, ast.ImportFrom)) for f in selected for n in ast.walk(f)): raise ValueError('Unexpected dynamic function imports')
 namespace = {'torch': torch, 'F': F, 'math': math, 'padded': padded}
 module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
 exec(compile(module, str(Path(source).resolve()), 'exec'), namespace)
 return namespace['teacher_inputs'], namespace['generalized_jsd']


def process_identity(pid):
 path = Path('/proc') / str(pid); stat = (path / 'stat').read_text(); fields = stat.rsplit(')', 1)[1].split()
 status = (path / 'status').read_text(); uidline = next(line for line in status.splitlines() if line.startswith('Uid:'))
 cgroup = (path / 'cgroup').read_text(); cmd = (path / 'cmdline').read_bytes()
 return {'pid': pid, 'start_ticks': fields[19], 'ppid': int(fields[1]), 'uids': [int(v) for v in uidline.split()[1:]],
 'raw_stat': stat, 'raw_status': status, 'raw_cgroup': cgroup, 'raw_cmdline_hex': cmd.hex()}


def owned_process(pid, job, uid):
 record = process_identity(pid)
 if record['uids'] != [uid] * 4 or not re.search(r'(?:^|/)job_' + re.escape(job) + r'(?:/|$)', record['raw_cgroup']): raise ValueError('Actual process not in own UID/Slurm allocation')
 if process_identity(pid)['start_ticks'] != record['start_ticks']: raise ValueError('Process reused during capture')
 return record


def hardware_guard(allocation, idle=False):
 """No model/index work before initial eligibility. Never signal other PIDs."""
 import fcntl
 import socket
 job = os.environ.get('SLURM_JOB_ID', ''); uid = os.getuid()
 if set(allocation) != {'version', 'job_id', 'uid', 'node', 'selected_gpu_uuid'} or allocation['version'] != 1: raise ValueError('Actual external allocation binding required')
 if not re.fullmatch('[0-9]+', job) or job in {'2994', '2996', '2999', '3007'} or allocation['job_id'] != job or allocation['uid'] != uid: raise ValueError('Fresh own Slurm job/UID required')
 if allocation['node'] != 'h100-10' or socket.gethostname().split('.')[0] != 'h100-10': raise ValueError('Only actual own H100 node permitted')
 terminal = subprocess.check_output(['scontrol', 'show', 'job', job], text=True)
 for key, value in [('JobId', job), ('JobState', 'RUNNING'), ('NodeList', 'h100-10'), ('NumNodes', '1'), ('TresPerNode', 'gres:gpu:h100:1')]:
  if re.findall(r'\b' + key + r'=([^\s]+)', terminal) != [value]: raise ValueError('Actual singleton H100 allocation differs')
 if re.findall(r'\bUserId=[^\s(]+\(([0-9]+)\)', terminal) != [str(uid)]: raise ValueError('External scheduler UID differs')
 selected = [v.strip() for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v.strip()]
 if len(selected) != 1: raise ValueError('Exactly one visible allocated CUDA device required')
 raw_gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
 matches = [r for r in csv.reader(raw_gpu.splitlines()) if selected[0] in (r[0].strip(), r[1].strip())]
 if len(matches) != 1 or matches[0][1].strip() != allocation['selected_gpu_uuid']: raise ValueError('Selected hardware UUID is not externally pinned')
 util, used, total = map(float, matches[0][2:])
 if not all(math.isfinite(v) for v in (util, used, total)) or not 0 <= util <= 100 or not 0 <= used <= total or total <= 0: raise ValueError('Invalid actual GPU occupancy')
 if idle and (util > 15 or used / total > .15): raise ValueError('Initial selected GPU exceeds original15-percent idle eligibility')
 leases = loads(os.environ.get('SEARCH_R1_GPU_LEASES', '[]'))
 if os.environ.get('SEARCH_R1_GPU_PREFLIGHT') != '1' or len(leases) != 1 or leases[0]['uuid'] != allocation['selected_gpu_uuid']: raise ValueError('Own inherited single-GPU lease required')
 lease_records = []
 for fd, path in [(9, ROOT / 'gpu-job.lock'), (int(leases[0]['fd']), Path('/tmp') / f'search-r1-gpu-leases-{uid}' / (allocation['selected_gpu_uuid'] + '.lock'))]:
  if fd != 9 and fd < 3 or int(leases[0]['fd']) == 9: raise ValueError('Lease descriptor overwritten or invalid')
  opened = os.fstat(fd); target = path.stat()
  if (opened.st_dev, opened.st_ino) != (target.st_dev, target.st_ino): raise ValueError('Inherited descriptor inode mismatch')
  fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  probe = os.open(path, os.O_RDWR)
  try:
   try: fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
   except BlockingIOError: pass
   else: raise ValueError('Independent descriptor unexpectedly acquired own lease')
  finally: os.close(probe)
  lease_records.append({'fd': fd, 'path': str(path), 'device': opened.st_dev, 'inode': opened.st_ino})
 raw_apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
 processes = []
 for record in csv.reader(raw_apps.splitlines()):
  if record and record[0].strip() == allocation['selected_gpu_uuid']:
   if len(record) != 3 or not record[1].strip().isdigit(): raise ValueError('Unreadable selected-GPU process identity')
   memory = float(record[2]);
   if not math.isfinite(memory) or memory < 0: raise ValueError('Invalid process GPU memory')
   processes.append(owned_process(int(record[1]), job, uid))
 return {'job_id': job, 'process': owned_process(os.getpid(), job, uid), 'slurm_raw': terminal, 'gpu_raw': raw_gpu,
 'apps_raw': raw_apps, 'gpu_uuid': allocation['selected_gpu_uuid'], 'leases': lease_records, 'selected_processes': processes}


def normalized_pci(value):
 match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value.strip())
 if not match: raise ValueError('Invalid actual PCI identity')
 return tuple(int(v, 16) for v in match.groups())


def cuda_actual_uuid():
 """Torch2.4 lacks properties.uuid: bind actual CUDA0 PCI to NVML UUID."""
 import ctypes
 import torch
 torch.cuda.cudart()  # initialize actual CUDA runtime only inside live collect
 libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libcudart.so' in line}
 if len(libraries) != 1: raise ValueError('Actual loaded CUDA runtime library ambiguous')
 library = ctypes.CDLL(next(iter(libraries)))
 function = library.cudaDeviceGetPCIBusId
 function.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]; function.restype = ctypes.c_int
 buffer = ctypes.create_string_buffer(64)
 if function(buffer, len(buffer), 0) != 0: raise ValueError('Actual CUDA0 PCI identity query failed')
 actual_pci = normalized_pci(buffer.value.decode('ascii'))
 raw = subprocess.check_output(['nvidia-smi', '--query-gpu=pci.bus_id,uuid', '--format=csv,noheader,nounits'], text=True)
 matches = [r[1].strip() for r in csv.reader(raw.splitlines()) if len(r)==2 and normalized_pci(r[0])==actual_pci]
 if len(matches)!=1: raise ValueError('Actual CUDA0 PCI to UUID mapping ambiguous')
 return matches[0], {'cuda0_pci': buffer.value.decode('ascii'), 'nvml_pci_uuid_raw': raw, 'loaded_libcudart': next(iter(libraries))}


def validate_runtime(m, allocation, initial=None):
 expected = [('base',['student_base'],'torch.bfloat16','flash_attention_2'), ('hf5',['student_hf5'],'torch.bfloat16','flash_attention_2'), ('base',['teacher_legacy','teacher_user_context'],'torch.float32','eager')]
 if m['runtime']['model_optimizer_steps'] != 0 or len(m['stages']) != 3: raise ValueError('Unexpected training dispatch/stage count')
 resources = [m['final_resource']] + ([] if initial is None else [initial])
 for stage, (role,cases,dtype,backend) in zip(m['stages'], expected):
  if (stage['role'],stage['cases'],stage['dtype'],stage['backend']) != (role,cases,dtype,backend): raise ValueError('Stage model role/dtype/backend drift')
  if stage['parameter_device'] != 'cuda:0' or stage['parameters_frozen'] is not True or stage['parameter_count'] != 3085938688 or stage['parameters_finite'] is not True or stage['tied_embeddings'] is not True: raise ValueError('Stage parameters not complete frozen CUDA model')
  if set(stage['loading']) != {'missing_keys','unexpected_keys','mismatched_keys','error_msgs'} or any(stage['loading'].values()): raise ValueError('Stage incomplete actual model load')
  if not 0 < stage['peak_reserved_bytes'] <= .85*stage['total_device_bytes']: raise ValueError('Stage allocator budget violated')
  if stage['cuda_mapping']['uuid'] != allocation['selected_gpu_uuid']: raise ValueError('Actual CUDA mapping drift')
  resources.append(stage['resource'])
 for resource in resources:
  if resource['job_id'] != allocation['job_id'] or resource['gpu_uuid'] != allocation['selected_gpu_uuid']: raise ValueError('Runtime allocation UUID drift')
  for process in [resource['process'],*resource['selected_processes']]:
   if process['uids'] != [allocation['uid']]*4 or not re.search(r'(?:^|/)job_'+re.escape(allocation['job_id'])+r'(?:/|$)',process['raw_cgroup']): raise ValueError('Runtime process ownership drift')
   uidline = next(line for line in process['raw_status'].splitlines() if line.startswith('Uid:'))
   if [int(v) for v in uidline.split()[1:]] != process['uids']: raise ValueError('Actual raw UID status differs')
   if process['raw_stat'].rsplit(')',1)[1].split()[19] != process['start_ticks']: raise ValueError('Runtime process start drift')


def selected_inputs(path, tokenizer):
 import pandas as pd
 frame = pd.read_parquet(path)
 if len(frame) < 64: raise ValueError('Training input too short')
 rows = []; seen = set()
 for i, (_, row) in enumerate(frame.iloc[:64].iterrows()):
  prompt = plain(row['prompt']); reward = plain(row['reward_model']); extra = plain(row['extra_info'])
  if row['data_source'] not in ('nq', 'hotpotqa') or extra.get('split') != 'train': raise ValueError('Only real train questions allowed')
  if not isinstance(prompt, list) or len(prompt) != 1 or prompt[0]['role'] != 'user': raise ValueError('Native single user prompt required')
  targets = reward['ground_truth']['target']; targets = [targets] if isinstance(targets, str) else targets
  if not targets or not all(isinstance(v, str) for v in targets): raise ValueError('Actual training target aliases missing')
  key = encoded(prompt)
  if key in seen: raise ValueError('Duplicate selected training question')
  seen.add(key)
  text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) if tokenizer.chat_template else prompt[0]['content']
  ids = tokenizer.encode(text, add_special_tokens=False)
  if not ids or len(ids) > 2048 or tokenizer.pad_token_id in ids: raise ValueError('Initial native prompt exceeds no-truncation bound or contains pad')
  rows.append({'ordinal': i, 'file_row': i, 'data_source': str(row['data_source']), 'raw_prompt': prompt, 'reward_model': reward,
   'extra_info': extra, 'student_prompt_text': text, 'student_prompt_ids': ids, 'targets': targets})
 return rows


def teacher_prefix(tokenizer, function, row, mode, maximum):
 import torch
 placeholder = tokenizer.encode('x', add_special_tokens=False)[0]
 if placeholder == tokenizer.pad_token_id: raise ValueError('Invalid audit placeholder')
 ids, attention, positions = function(tokenizer, torch.tensor([row['student_prompt_ids']]), torch.tensor([[placeholder]]), torch.ones(1, 1, dtype=torch.long),
   [row['targets']], maximum, prompt_mode=mode)
 length = int(positions[0, 0]) + 1
 if length < 1 or int(ids[0, length]) != placeholder or int(attention.sum()) != length + 1: raise ValueError('Teacher actual initial prediction position does not align')
 return ids[0, :length].tolist()


def objective_metrics(student, teacher, function):
 import torch
 s = torch.as_tensor(student, dtype=torch.float32).clone().requires_grad_(True); t = torch.as_tensor(teacher, dtype=torch.float32)
 if s.ndim != 1 or s.shape != t.shape or not torch.isfinite(s).all() or not torch.isfinite(t).all(): raise ValueError('Full-vocabulary finite logits required')
 unclipped = function(s, t, 0., 1., None); clipped = function(s, t, 0., 1., .05)
 gu, = torch.autograd.grad(unclipped, s, retain_graph=True); gc_, = torch.autograd.grad(clipped, s)
 lp_s, lp_t = s.detach().log_softmax(-1), t.log_softmax(-1); p_t = lp_t.exp(); components = p_t * (lp_t - lp_s); active = components > .05
 nu, nc = float(gu.norm()), float(gc_.norm()); cosine = None if nu == 0 or nc == 0 else float(torch.dot(gu, gc_) / (gu.norm() * gc_.norm()))
 result = {'unclipped_kl': float(unclipped.detach()), 'component_clipped_loss': float(clipped.detach()), 'clipped_components': int(active.sum()),
 'clipped_vocab_fraction': float(active.float().mean()), 'clipped_teacher_probability_mass': float(p_t[active].sum()),
 'clipped_positive_component_mass': float(components[active].sum()), 'gradient_cosine': cosine,
 'unclipped_logit_gradient_norm': nu, 'clipped_logit_gradient_norm': nc, 'hypothetical_logit_step_l2': .01,
 'meaning': 'CPU logit-space direction only; not a model/optimizer update or measured search effect'}
 for name, grad, norm in [('unclipped', gu, nu), ('clipped', gc_, nc)]:
  updated = s.detach() if norm == 0 else s.detach() - .01 * grad / norm
  after = float(function(updated, t, 0., 1., None))
  result[name + '_step_unclipped_kl'] = after; result[name + '_step_unclipped_kl_delta'] = after - result['unclipped_kl']
 return plain(result)


def forward_case(model, tokenizer, prefix, path, case, row):
 import numpy as np
 import torch
 from contextlib import nullcontext
 teacher = case.startswith('teacher_'); arrays = {'prompt_ids': np.asarray(prefix, dtype=np.int64)}
 def forward(ids, start):
  input_ids = torch.tensor([ids], dtype=torch.long, device='cuda:0'); mask = torch.ones_like(input_ids)
  with torch.inference_mode(), (nullcontext() if teacher else torch.autocast('cuda', dtype=torch.bfloat16)):
   logits = model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits
  return logits[0, start:].float().cpu()
 initial = forward(prefix, len(prefix)-1)[0]; arrays['initial_logits'] = initial.numpy(); tag_metrics = {}
 candidates = (*TAGS, '<answer>' + row['targets'][0] + '</answer>')
 for index, tag in enumerate(candidates):
  tokens = tokenizer.encode(tag, add_special_tokens=False)
  if not tokens or any(v in tokenizer.all_special_ids for v in tokens): raise ValueError('Tag candidate contains control tokens')
  if len(prefix) + len(tokens) - 1 > model.config.max_position_embeddings:
   if index < 3: raise ValueError('Required tag exceeds context limit')
   tag_metrics[tag] = {'skipped': True, 'reason': 'complete accepted-answer candidate exceeds actual context limit', 'tokens': tokens}
   arrays[f'tag{index}_ids'] = np.asarray(tokens, dtype=np.int64)
   continue
  logits = forward(prefix + tokens[:-1], len(prefix)-1)
  if logits.shape != (len(tokens), model.config.vocab_size): raise ValueError('Full tag conditional positions mismatch')
  arrays[f'tag{index}_ids'] = np.asarray(tokens, dtype=np.int64); arrays[f'tag{index}_logits'] = logits.numpy()
  values = logits.log_softmax(-1).gather(-1, torch.tensor(tokens)[:, None]).squeeze(-1)
  tag_metrics[tag] = {'conditional_log_probability': float(values.sum()), 'tokens': tokens, 'token_log_probabilities': values.tolist()}
 if any(not np.isfinite(v).all() for k, v in arrays.items() if 'logits' in k): raise ValueError('Nonfinite actual model logits')
 with path.open('xb') as handle: np.savez(handle, **arrays)
 return {'case': case, 'ordinal': row['ordinal'], 'path': str(path), 'sha256': digest(path), 'vocab_size': model.config.vocab_size,
  'tags': tag_metrics, 'scope': 'next-token/full-tag/accepted-answer candidate conditional logprob; NOT completion/accuracy'}


def replay(directory, expected):
 import numpy as np
 import torch
 directory = Path(directory).resolve(); manifest = bound({'path': str(directory / 'manifest.json'), 'sha256': expected}, directory)
 m = loads(manifest.read_bytes())
 if m['scope'] != 'train-initial-state-distribution-only' or len(m['rows']) != 64 or len(m['cases']) != 256: raise ValueError('Wrong diagnostic scope/count')
 verify_binding(m['binding'])
 if m['source'] != m['binding']['source'] or m['probe_sha256'] != m['binding']['probe_sha256']: raise ValueError('Source identity drift')
 source = bound(m['source']); _, objective = extract_functions(source)
 if loads(bound(m['binding_file'],directory).read_bytes()) != m['binding']: raise ValueError('Collected external binding changed')
 allocation = loads(bound(m['allocation'],directory).read_bytes())
 initial_resource = loads(bound(m['initial_resource'],directory).read_bytes())
 if initial_resource['job_id'] != allocation['job_id'] or m['final_resource']['job_id'] != allocation['job_id'] or m['final_resource']['gpu_uuid'] != allocation['selected_gpu_uuid']: raise ValueError('Actual allocation/resource identity differs')
 validate_runtime(m,allocation,initial_resource)
 inputs = loads(bound(m['inputs'], directory).read_bytes())
 if len(inputs) != 64 or m['rows'] != [hashlib.sha256(encoded(r)).hexdigest() for r in inputs]: raise ValueError('Input identity/count differs')
 cases = {}; teacher_fn, _ = extract_functions(source)
 from transformers import AutoTokenizer
 tokenizer = AutoTokenizer.from_pretrained(m['binding']['models']['base']['directory'], local_files_only=True)
 if encoded(selected_inputs(TRAIN, tokenizer)) != encoded(inputs): raise ValueError('Output input identity not authentic original first64 training rows')
 config = loads((BASE/'config.json').read_bytes()); vocabulary = config['vocab_size']; maximum = config['max_position_embeddings']
 for record in m['cases']:
  key = (record['ordinal'], record['case'])
  if key in cases or not 0 <= key[0] < 64 or key[1] not in CASE_NAMES: raise ValueError('Duplicate/unknown diagnostic case')
  path = bound({'path': record['path'], 'sha256': record['sha256']}, directory)
  with np.load(path, allow_pickle=False) as data:
   arrays = {k: data[k].copy() for k in data.files}
  expected_prefix = inputs[key[0]]['student_prompt_ids'] if key[1].startswith('student_') else teacher_prefix(tokenizer, teacher_fn, inputs[key[0]], key[1].removeprefix('teacher_'), 32768)
  if arrays['prompt_ids'].tolist() != expected_prefix: raise ValueError('Actual prompt IDs not prescribed student/teacher context')
  if record['vocab_size'] != vocabulary or arrays['initial_logits'].dtype != np.float32 or not np.isfinite(arrays['initial_logits']).all() or arrays['initial_logits'].shape != (vocabulary,): raise ValueError('Saved vocabulary truncated')
  expected_keys = {'prompt_ids', 'initial_logits'}
  for i, tag in enumerate((*TAGS, '<answer>'+inputs[key[0]]['targets'][0]+'</answer>')):
   expected_keys.add(f'tag{i}_ids')
   if arrays[f'tag{i}_ids'].tolist() != tokenizer.encode(tag, add_special_tokens=False): raise ValueError('Candidate token identity changed')
   if len(expected_prefix)+len(arrays[f'tag{i}_ids'])-1 > maximum:
    if i < 3 or record['tags'][tag] != {'skipped': True, 'reason': 'complete accepted-answer candidate exceeds actual context limit', 'tokens': arrays[f'tag{i}_ids'].tolist()}: raise ValueError('Invalid candidate skip')
    continue
   expected_keys.add(f'tag{i}_logits')
   if arrays[f'tag{i}_logits'].dtype != np.float32: raise ValueError('Conditional logits dtype changed')
   tokens = arrays[f'tag{i}_ids']; logits = torch.from_numpy(arrays[f'tag{i}_logits'])
   if tokens.tolist() != tokenizer.encode(tag, add_special_tokens=False) or logits.shape != (len(tokens), record['vocab_size']) or not torch.isfinite(logits).all(): raise ValueError('Tag tokens/full distribution changed')
   values = logits.log_softmax(-1).gather(-1, torch.tensor(tokens)[:, None]).squeeze(-1)
   if encoded(record['tags'][tag]) != encoded({'conditional_log_probability': float(values.sum()), 'tokens': tokens.tolist(), 'token_log_probabilities': values.tolist()}): raise ValueError('Tag online/replayed metrics differ')
  if set(arrays) != expected_keys: raise ValueError('Raw scoring view keys missing or surplus')
  cases[key] = arrays['initial_logits']
 rows = []
 for i in range(64):
  if {(i, case) for case in CASE_NAMES} - set(cases): raise ValueError('Missing initial-state case')
  for student in ('student_base', 'student_hf5'):
   for teacher in ('teacher_legacy', 'teacher_user_context'):
    rows.append({'ordinal': i, 'student': student, 'teacher': teacher, 'metrics': objective_metrics(cases[i, student], cases[i, teacher], objective)})
 aggregates = []
 for student in ('student_base', 'student_hf5'):
  for teacher in ('teacher_legacy', 'teacher_user_context'):
   matched = [r['metrics'] for r in rows if r['student']==student and r['teacher']==teacher]
   aggregates.append({'student':student,'teacher':teacher,'count':64,'rows_with_component_clipping':sum(r['clipped_components']>0 for r in matched),'rows_with_opposed_logit_gradient':sum(r['gradient_cosine'] is not None and r['gradient_cosine']<0 for r in matched),'rows_with_clipped_step_kl_increase':sum(r['clipped_step_unclipped_kl_delta']>0 for r in matched),'mean_unclipped_kl':sum(r['unclipped_kl'] for r in matched)/64,'mean_clipped_loss':sum(r['component_clipped_loss'] for r in matched)/64})
 return {'scope': m['scope'], 'manifest_sha256': expected, 'questions': 64, 'comparison_rows': rows, 'aggregates':aggregates, 'conditional_candidate_rows':[{'ordinal':r['ordinal'],'case':r['case'],'conditional_candidates':r['tags']} for r in m['cases']], 'limitations': m['limitations']}


def collect(args):
 import torch
 from transformers import AutoTokenizer, AutoModelForCausalLM
 binding = loads(bound({'path': args.binding, 'sha256': args.binding_sha256}).read_bytes()); verify_binding(binding)
 allocation = loads(bound({'path': args.allocation, 'sha256': args.allocation_sha256}).read_bytes())
 initial = hardware_guard(allocation, idle=True)
 if torch.cuda.device_count() != 1: raise ValueError('Actual PyTorch visibility not singleton')
 gpu_uuid, cuda_mapping = cuda_actual_uuid()
 if gpu_uuid != allocation['selected_gpu_uuid']: raise ValueError('Actual torch device UUID not selected hardware')
 torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
 directory = Path(args.output)
 if not directory.is_absolute() or not directory.resolve().is_relative_to(ROOT) or directory.exists(): raise ValueError('New immutable own output directory required')
 directory.mkdir(parents=True)
 args._owned_output = directory.resolve(); args._owner_job = allocation['job_id']; args._owner_uid = allocation['uid']
 write_new(directory / 'binding.json', binding); write_new(directory / 'allocation.json', allocation)
 write_new(directory / 'initial-resource.json', initial)
 tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
 rows = selected_inputs(TRAIN, tokenizer); write_new(directory / 'inputs.json', rows)
 teacher_fn, _ = extract_functions(binding['source']['path']); records = []; stages = []
 for role, cases, dtype in [('base', ('student_base',), torch.bfloat16), ('hf5', ('student_hf5',), torch.bfloat16), ('base', ('teacher_legacy', 'teacher_user_context'), torch.float32)]:
  torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); hardware_guard(allocation)
  model_path = binding['models'][role]['directory']; model, loading = AutoModelForCausalLM.from_pretrained(model_path, output_loading_info=True, torch_dtype=dtype, attn_implementation='eager' if dtype == torch.float32 else 'flash_attention_2', local_files_only=True)
  if set(loading) != {'missing_keys','unexpected_keys','mismatched_keys','error_msgs'} or any(loading.values()): raise ValueError('Actual model load is incomplete/mismatched')
  base_config = loads((BASE/'config.json').read_bytes())
  if any(getattr(model.config,k) != base_config[k] for k in ('model_type','vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','max_position_embeddings')): raise ValueError('Actual HF5/base architecture differs')
  model = model.to('cuda:0').eval().requires_grad_(False)
  tied_embeddings = model.config.tie_word_embeddings is True and model.get_output_embeddings().weight is model.get_input_embeddings().weight
  if not tied_embeddings: raise ValueError('Actual fixed Qwen model embedding tie differs')
  parameters_finite = all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
  if not parameters_finite or sum(parameter.numel() for parameter in model.parameters()) != 3085938688: raise ValueError('Actual model parameter count/content is incomplete or nonfinite')
  if any(p.requires_grad or p.grad is not None or p.dtype != dtype or p.device.type != 'cuda' for p in model.parameters()): raise ValueError('Model inference dtype/freeze/device differs')
  if tokenizer.get_vocab() != AutoTokenizer.from_pretrained(model_path, local_files_only=True).get_vocab(): raise ValueError('Actual student/teacher token IDs differ')
  for row in rows:
   for case in cases:
    prefix = row['student_prompt_ids'] if case.startswith('student_') else teacher_prefix(tokenizer, teacher_fn, row, case.removeprefix('teacher_'), model.config.max_position_embeddings)
    record = forward_case(model, tokenizer, prefix, directory / f"row{row['ordinal']:03d}-{case}.npz", case, row); records.append(record)
    write_new(directory / f"row{row['ordinal']:03d}-{case}.json", record)
   if row['ordinal'] % 8 == 0: hardware_guard(allocation)
  torch.cuda.synchronize(); peak = torch.cuda.max_memory_reserved(); total = torch.cuda.get_device_properties(0).total_memory
  if peak > .85 * total: raise ValueError('Actual allocator reserve exceeds85-percent headroom gate')
  if model.config._attn_implementation != ('eager' if dtype == torch.float32 else 'flash_attention_2'): raise ValueError('Actual attention backend differs')
  stages.append({'loading': plain(loading), 'backend':model.config._attn_implementation, 'parameter_count':sum(p.numel() for p in model.parameters()),'parameters_finite':parameters_finite,'tied_embeddings':tied_embeddings,'parameters_frozen':all(not p.requires_grad and p.grad is None for p in model.parameters()),'parameter_device':'cuda:0','cuda_mapping':{'uuid':gpu_uuid,**cuda_mapping},'total_device_bytes':total,'role': role, 'cases': list(cases), 'dtype': str(dtype), 'peak_reserved_bytes': peak, 'resource': hardware_guard(allocation)})
  del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
 verify_binding(binding)  # original model/data/source complete contents remain unchanged
 m = {'version': 1, 'scope': binding['scope'], 'binding': binding, 'source': binding['source'], 'probe_sha256': digest(Path(__file__)),
  'inputs': {'path': str(directory / 'inputs.json'), 'sha256': digest(directory / 'inputs.json')},
  'rows': [hashlib.sha256(encoded(r)).hexdigest() for r in rows], 'cases': records, 'stages': stages, 'final_resource': hardware_guard(allocation),
  'allocation': {'path': str(directory/'allocation.json'), 'sha256': digest(directory/'allocation.json')},
  'binding_file': {'path': str(directory/'binding.json'), 'sha256': digest(directory/'binding.json')},
  'initial_resource': {'path': str(directory/'initial-resource.json'), 'sha256': digest(directory/'initial-resource.json')},
  'runtime': {'python': sys.version, 'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'model_optimizer_steps': 0},
  'limitations': ['64 ordered TRAIN INITIAL STATES only; not late invalid histories', 'Tag or accepted-answer candidate logprob is not generated valid action/completion/EM/F1', 'HF step5 only; no step80/500 weights exist', 'No retrieval or optimizer; CPU gradients are logit-space hypothetical steps only', 'Sequential HF inference, not FSDP training-forward numerical equivalence']}
 write_new(directory / 'manifest.json', m)
 # Native CPU replay can run separately after this job; do not claim terminal/quality receipt.
 print('OPSD_INITIAL_DISTRIBUTION_MANIFEST_SHA256=' + digest(directory / 'manifest.json'), flush=True)


def self_test(source):
 import torch
 _, function = extract_functions(source)
 student = torch.tensor([.1, .9]).log(); teacher = torch.tensor([.9, .1]).log()
 result = objective_metrics(student, teacher, function)
 assert result['gradient_cosine'] < -.99 and result['clipped_step_unclipped_kl_delta'] > 0 and result['unclipped_step_unclipped_kl_delta'] < 0
 assert result['clipped_components'] == 1 and abs(result['clipped_teacher_probability_mass'] - .9) < 1e-6
 assert objective_metrics(teacher, teacher, function)['unclipped_kl'] == 0
 for invalid in [torch.tensor([float('nan'), 0.]), torch.ones(3)]:
  try: objective_metrics(invalid, teacher, function)
  except ValueError: pass
  else: raise AssertionError('Bad distribution accepted')
 print(json.dumps({'scope': 'CPU function/logit mathematics only; NOT model/GPU effect proof', 'source_sha256': digest(source), 'counterexample': result}, sort_keys=True))


def preserve_failure(args, error):
 """Only this invocation's exclusively-created directory may receive a marker."""
 directory = getattr(args, '_owned_output', None)
 if directory is None: return
 if getattr(args, '_owner_job', None) != os.environ.get('SLURM_JOB_ID') or getattr(args, '_owner_uid', None) != os.getuid(): return
 directory = Path(directory)
 if directory.is_absolute() and directory.resolve().is_relative_to(ROOT) and directory.is_dir() and not (directory/'failure.json').exists():
  write_new(directory/'failure.json', {'scope':'FAILED diagnostic, not completed evidence','error_type':type(error).__name__,'error':str(error),'job_id':args._owner_job,'probe_sha256':digest(Path(__file__))})


def main():
 parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='mode', required=True)
 cpu = commands.add_parser('cpu-self-test'); cpu.add_argument('--source', required=True)
 replay_cmd = commands.add_parser('replay'); replay_cmd.add_argument('--directory', required=True); replay_cmd.add_argument('--manifest-sha256', required=True); replay_cmd.add_argument('--output', required=True)
 live = commands.add_parser('collect'); live.add_argument('--binding', required=True); live.add_argument('--binding-sha256', required=True); live.add_argument('--allocation', required=True); live.add_argument('--allocation-sha256', required=True); live.add_argument('--output', required=True)
 args = parser.parse_args()
 if args.mode == 'cpu-self-test': self_test(args.source)
 elif args.mode == 'replay':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''; write_new(args.output, replay(args.directory, args.manifest_sha256))
 else:
  try: collect(args)
  except BaseException as error:
   preserve_failure(args,error)
   raise


if __name__ == '__main__': main()
