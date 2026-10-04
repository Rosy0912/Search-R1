"""Create a versioned experiment overlay without changing the verified archive."""
from pathlib import Path
import argparse,shutil,json,hashlib,difflib
from optimizer_chunks import patch_optimizer_chunks
from token_statistics import patch_statistics
from physical_micro import patch_actor
from opsd_execution import patch_opsd
from jit_optimizer import patch_actor_optimizer,patch_worker_optimizer
p=argparse.ArgumentParser();p.add_argument('--purpose',choices=['grpo','opsd'],default='grpo');p.add_argument('--base',required=True);p.add_argument('--profile',choices=['archive','r1'],required=True);a=p.parse_args()
base=Path(a.base);repo=base/'source';archive=repo/'experiments/search_e1/archive';runtime=base/('runtime' if a.purpose=='grpo' else 'opsd-runtime')/a.profile
receipt_profile=a.profile if a.purpose=='grpo' else a.profile+'-opsd'
if runtime.exists():raise SystemExit('Runtime already exists; refusing to overwrite frozen source')
shutil.copytree(archive,runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.cache'))
shutil.copyfile(Path(__file__).with_name('virtual_dp.py'), runtime/'search_e1/virtual_dp.py')
up=runtime/'upstream/Search-R1';shutil.copyfile(repo/'search_r1/qa_protocol.py',up/'search_r1/qa_protocol.py')
legacy=up/'verl/utils/reward_score/qa_em.py';shutil.copyfile(legacy,runtime/'search_e1/historical_qa_em.py')
s=legacy.read_text();s+='''\nfrom search_r1.qa_protocol import extract_final_answer

def compute_score_generated_em(model_spans, ground_truth, format_score=0., score=1.):
    answer = extract_final_answer(model_spans)
    if answer is None:
        return 0.
    return score if em_check(answer, ground_truth['target']) else format_score
''';legacy.write_text(s)
p=up/'verl/trainer/main_ppo.py';s=p.read_text();s=s.replace('from verl import DataProto','from verl import DataProto\nfrom search_r1.qa_protocol import decode_model_spans\nfrom search_e1 import historical_qa_em',1).replace('return qa_em.compute_score_em','return qa_em.compute_score_generated_em',1)
s=s.replace('already_print_data_sources = {}','already_print_data_sources = {}\n        old_scores, new_scores = [], []',1)
s=s.replace("valid_response_ids = response_ids[:valid_response_length]",'''if int(valid_response_length) == 0:
                old_scores.append(0.); new_scores.append(0.)
                continue
            valid_response_ids = response_ids[:valid_response_length]
            if 'info_mask' not in data_item.batch:
                raise ValueError('Search-E1 search rollouts require info_mask provenance')
            model_mask = data_item.batch['info_mask'][prompt_length:prompt_length + int(valid_response_length)]
            model_spans = decode_model_spans(self.tokenizer, valid_response_ids, model_mask)''',1)
s=s.replace('score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)', '''score = compute_score_fn(model_spans=model_spans, ground_truth=ground_truth, format_score=self.format_score)
            old_scores.append(float(historical_qa_em.compute_score_em(sequences_str, ground_truth, format_score=self.format_score)))
            new_scores.append(float(score))''',1)
s=s.replace('        return reward_tensor','''        self.last_metrics = {'audit/legacy_em': float(np.mean(old_scores)),
                             'audit/strict_em': float(np.mean(new_scores)),
                             'audit/score_disagreement': float(np.mean(np.array(old_scores) != np.array(new_scores)))}
        return reward_tensor''',1);p.write_text(s)
p=up/'verl/trainer/ppo/ray_trainer.py';s=p.read_text();s=s.replace('reward_tensor = self.reward_fn(batch)','reward_tensor = self.reward_fn(batch)\n                        metrics.update(getattr(self.reward_fn, "last_metrics", {}))',1)
# A short smoke must retain its full run LR schedule; stop_after only limits execution.
needle="                if self.global_steps >= self.total_training_steps:"
assert needle in s
s=s.replace(needle,'''                stop_after = self.config.trainer.get('stop_after', None)
                if stop_after is not None and self.global_steps - 1 >= int(stop_after):
                    print('SMOKE_STOP_AFTER_REAL_UPDATES', self.global_steps - 1, flush=True)
                    return
''' + needle,1);p.write_text(s)
# Preserve the original four-way token reduction when B uses eight GPUs.
p=up/'verl/trainer/ppo/ray_trainer.py';s=p.read_text()
needle='        world_size = self.actor_rollout_wg.world_size'
assert s.count(needle)==1
s=s.replace(needle, '''        physical_world = self.actor_rollout_wg.world_size
        virtual_world = int(os.environ.get('E1_VIRTUAL_DP_WORLD', '0'))
        if virtual_world and (virtual_world != 4 or physical_world != 8):
            raise ValueError('Virtual DP requires four logical/eight physical workers')
        world_size = virtual_world or physical_world''',1)
needle='        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])'
assert s.count(needle)==1
s=s.replace(needle, '''        if virtual_world:
            from search_e1.virtual_dp import stripe_partitions
            indices = stripe_partitions(global_partition_lst, logical_micro=16)
        else:
            indices = [j for partition in global_partition_lst for j in partition]
        global_idx = torch.tensor(indices)''',1);p.write_text(s)
p=up/'verl/workers/actor/dp_actor.py';s=p.read_text();s=s.replace('        self.actor_optimizer.step()', '''        if not torch.isfinite(grad_norm).all():
            raise FloatingPointError('Nonfinite actor gradient; optimizer update refused')
        self.actor_optimizer.step()''',1).replace('                loss.backward()', '''                if not torch.isfinite(loss).all():
                    raise FloatingPointError('Nonfinite actor loss; update refused')
                loss.backward()''',1);p.write_text(s)
p=up/'verl/workers/actor/dp_actor.py';p.write_text(patch_optimizer_chunks(patch_statistics(patch_actor_optimizer(patch_actor(p.read_text())))))
p=up/'verl/workers/fsdp_workers.py';p.write_text(patch_worker_optimizer(p.read_text()))
p=runtime/'search_e1/core.py';s=p.read_text();start=s.index('def reward(');end=s.index('\ndef ',start+5);s=s[:start]+'''def reward(trajectory, answers, policy_spans=None):
    from search_r1.qa_protocol import extract_final_answer
    if policy_spans is None:
        raise ValueError('Policy span provenance is required for Search-E1 rewards')
    previous = 0
    spans = []
    for start, end in policy_spans:
        if not previous <= start <= end <= len(trajectory):
            raise ValueError('Invalid policy span bounds')
        spans.append(trajectory[start:end]); previous = end
    answer = extract_final_answer(spans)
    if answer is None:
        return 0
    if isinstance(answers, str):
        answers = [answers]
    return int(any(normalize(answer) == normalize(a) for a in answers))

'''+s[end:];p.write_text(s)
p=runtime/'search_e1/rollout.py';s=p.read_text();s=s.replace("values = [reward(agent.run(r['question'], temperature=0)['response'], r['answers']) for r in rows]",'''values = []
    for row in rows:
        result = agent.run(row['question'], temperature=0)
        values.append(reward(result['response'], row['answers'], result['policy_spans']))''').replace("reward(result['response'], result['answers'])","reward(result['response'], result['answers'], result['policy_spans'])");p.write_text(s)
# Standalone inference has no resident FSDP actor: tune separately, keep reserve.
p=runtime/'search_e1/vllm_rollout.py';s=p.read_text();s=s.replace('import re', 'import re\nimport os',1).replace('gpu_memory_utilization=0.6,', "gpu_memory_utilization=float(os.environ.get('E1_INFERENCE_UTIL', '0.85')),");p.write_text(s)
# The B arm must not inherit trajectory batching just because microbatch > 1.
p=runtime/'scripts/grpo.sh';s=p.read_text();s=s.replace('if (( MICRO_PER_GPU > 1 )); then', 'if (( MICRO_PER_GPU > 1 )) && [[ "${E1_DISABLE_TRAJECTORY_BATCHING:-0}" != 1 ]]; then');p.write_text(s)
# Permit the shared same-cluster exact CPU index service to encode on CPU.
p=up/'search_r1/search/retrieval_server.py';s=p.read_text();s='import os\n'+s
s=s.replace('model.cuda()', "model.to(os.environ.get('E1_ENCODER_DEVICE', 'cuda'))")
s=s.replace('v.cuda()', 'v.to(next(self.model.parameters()).device)');p.write_text(s)
# Always start a task-private Ray cluster, independent of any ambient RAY_ADDRESS.
p=runtime/'search_e1/grpo_main.py';s=p.read_text();s=s.replace("    job = os.environ.get('SLURM_JOB_ID', str(os.getpid()))", "    for key in ['E1_PHYSICAL_MICRO', 'E1_PHYSICAL_SCORE', 'E1_JIT_OPTIMIZER', 'E1_LOGIT_CHUNK', 'E1_OPTIMIZER_CHUNK','E1_STAGE_CHUNK_STATES', 'E1_VIRTUAL_DP_WORLD']:\n        if key in os.environ:\n            kwargs['runtime_env']['env_vars'][key] = os.environ[key]\n    job = os.environ.get('SLURM_JOB_ID', str(os.getpid()))",1);s=s.replace("kwargs.setdefault('num_cpus'", "kwargs['address'] = 'local'\n    kwargs['_system_config'] = dict(kwargs.get('_system_config', {}), enable_metrics_collection=False)\n    kwargs.setdefault('num_cpus'",1).replace("job = os.environ.get('SLURM_JOB_ID', str(os.getpid()))", "job = os.environ.get('E1_RUN_ID', str(os.getpid()))",1).replace("kwargs.setdefault('_temp_dir', '/tmp/search-e1-ray-' + job)", "real_temp = Path(os.environ['E1_BASE']) / 'ray' / job\n    real_temp.mkdir(parents=True, exist_ok=True)\n    short_temp = Path('/tmp') / ('e1-' + str(os.getuid()) + '-' + job[:8])\n    if short_temp.exists() or short_temp.is_symlink():\n        if short_temp.resolve() != real_temp.resolve():\n            raise RuntimeError('Ray temporary link collision')\n    else:\n        short_temp.symlink_to(real_temp, target_is_directory=True)\n    kwargs.setdefault('_temp_dir', str(short_temp))",1).replace("    return main_ppo.main.__wrapped__(config)", "    try:\n        return main_ppo.main.__wrapped__(config)\n    finally:\n        ray.shutdown()",1);p.write_text(s)
# Match driver shuffle and policy sampler seeds in both diagnostic arms.
p=up/'verl/trainer/main_ppo.py';s=p.read_text();s=s.replace('def main_task(config):', 'def main_task(config):\n    import random\n    random.seed(42); np.random.seed(42); torch.manual_seed(42)',1);p.write_text(s)
p=up/'verl/workers/rollout/vllm_rollout/vllm_rollout.py';s=p.read_text();s=s.replace('                                    dtype=config.dtype,', '                                    dtype=config.dtype,\n                                    seed=42,',1);p.write_text(s)
# Separate OPSD runtime adds collective rejection and distributed heldout EM.
if a.purpose == 'opsd':
 p=runtime/'search_e1/opsd.py';p.write_text(patch_opsd(p.read_text()))
# Use fixed local dataset revision, not a fresh online load during each arm.
p=runtime/'search_e1/prepare.py';s=p.read_text();s=s.replace('from datasets import load_dataset, Dataset','from datasets import load_dataset, Dataset\nimport os');s=s.replace("ds = load_dataset('RUC-NLPIR/FlashRAG_datasets', source)","files = {p.stem: str(p) for p in (Path(os.environ['E1_DATASET_ROOT']) / source).glob('*.jsonl')}\n        if not files:\n            raise ValueError('No pinned dataset files for ' + source)\n        ds = load_dataset('json', data_files=files)");p.write_text(s)
patch=[];manifest={}
for path in runtime.rglob('*'):
 if not path.is_file():continue
 rel=path.relative_to(runtime);old=archive/rel
 if not old.exists() or old.read_bytes()!=path.read_bytes():
  if path.suffix in ['.py','.sh','.yaml']:
   patch.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],path.read_text().splitlines(True),fromfile='archive/'+str(rel),tofile='runtime/'+str(rel)))
 manifest[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
(base/'receipts').mkdir(exist_ok=True)
(base/'receipts'/f'{receipt_profile}-runtime.sha256.json').write_text(json.dumps(manifest,indent=2))
(base/'receipts'/f'{receipt_profile}-overlay.patch').write_text(''.join(patch))
print(runtime)
