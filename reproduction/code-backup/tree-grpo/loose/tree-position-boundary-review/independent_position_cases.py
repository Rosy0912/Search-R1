"""Independent source/CPU probes: no real calibration assets or remote calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

import torch

WORK = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
EXPECTED = '3992c092e7cf4e42469b9e2cf7cab896a44395c70f224101ef37cb4c0fee05c9'
EXPECTED_TEST = '6f33ed1380f96bc3e77e389ff87cd2a8f4970aada5963ca9887ecbff50f48fa2'

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert digest(WORK/'inspect_tree_position_boundaries.py') == EXPECTED
assert digest(WORK/'test_inspect_tree_position_boundaries.py') == EXPECTED_TEST
for path in [WORK/'inspect_tree_position_boundaries.py', WORK/'test_inspect_tree_position_boundaries.py']:
    compile(path.read_bytes(), str(path), 'exec')
    destination = OUT/('candidate-'+digest(path)+'.py')
    if not destination.exists(): shutil.copyfile(path, destination)
    assert digest(destination) == digest(path)
spec = importlib.util.spec_from_file_location('position_owner_fixtures', WORK/'test_inspect_tree_position_boundaries.py')
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
P = T.P
A = P.context_tools('28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac')
torch.set_num_threads(4)
started = time.perf_counter(); cpu_started = time.process_time()
checks = []

def record(name, evidence=None): checks.append(dict(name=name, evidence=evidence))
def refused(call, name):
    try: call()
    except (ValueError, KeyError, StopIteration): record(name); return
    raise AssertionError('Admitted invalid evidence: '+name)

with tempfile.TemporaryDirectory(prefix='position-independent-', dir=OUT) as directory:
    root = Path(directory)
    hf = root/'hf.py'; hf.write_text(T.HF_FIXTURE)
    rules = P.hf_rules(hf, P.sha(hf))
    out = P.inspect_positions(T.inputs([[0,1],[90,91],[0,1]], mask=[[1,1],[0,0],[1,1]]), rules, 'empty-middle')
    assert out['rows']['empty_rows'] == 1
    assert all(v['missing_boundaries'] == v['extra_boundaries'] == 0 for v in out['local_grouping'].values())
    record('empty_intermediate_rows_have_no_false_positive_boundaries')

    out = P.inspect_positions(T.inputs([[3,2],[1,1]]), rules, 'no-zero-descending')
    stats = out['local_grouping']['4']
    assert stats['packed_varlen_groups'] == stats['groups_with_uncovered_prefix'] == 1
    assert stats['possible_cross_sample_boundaries'] == 0 and stats['missing_sample_boundaries'] == 1
    assert 'absence of counted crossing segments does not establish valid attention' in out['limits']
    record('no_zero_packed_path_has_uncovered_prefix_even_when_crossing_counter_zero', stats)

    values = T.inputs([[91,0,1,99],[91,0,1,99]], mask=[[0,1,1,0]]*2, loss=[[0,0,1,0]]*2)
    before = P.inspect_positions(values, rules, 'padding')
    values['position_ids'][:,0] += 100
    values['input_ids'][:,-1] += 100
    after = P.inspect_positions(values, rules, 'padding')
    assert before['rows'] == after['rows'] and before['local_grouping'] == after['local_grouping']
    assert before['full_input_tensor_identity'] != after['full_input_tensor_identity']
    record('masked_padding_changes_full_bytes_identity_without_changing_attention_segmentation')

    values = T.inputs([[0,1],[0,1]], loss=[[0,0],[0,0]])
    out = P.inspect_positions(values, rules, 'all-loss-zero')
    assert out['rows']['attention_valid_tokens'] == 4 and out['local_grouping']['4']['missing_boundaries'] == 0
    record('all_loss_mask_zero_still_preserves_full_attention_observation_context')
    bad = T.inputs([[0,1]]); bad['position_ids'] = bad['position_ids'].double()
    refused(lambda:P.inspect_positions(bad,rules,'float'), 'floating_positions_refused')
    bad = T.inputs([[0,1]]); bad['attention_mask'] = torch.tensor([[1.,float('nan')]])
    refused(lambda:P.inspect_positions(bad,rules,'nan'), 'nan_attention_mask_refused')
    bad = T.inputs([[0,1]]); bad['input_ids'] = torch.tensor([0,1])
    refused(lambda:P.inspect_positions(bad,rules,'shape'), 'full_context_shape_mismatch_refused')

    original = T.inputs([[0,1],[2,3],[0,1],[2,3]])
    reordered = {key:value[[0,2,1,3]] for key,value in original.items()}
    first = P.inspect_positions(original,rules,'pre-balance')['local_grouping']['4']
    second = P.inspect_positions(reordered,rules,'post-balance')['local_grouping']['4']
    assert first['attention_valid_tokens_in_crossing_segments'] != second['attention_valid_tokens_in_crossing_segments']
    record('same_rows_reordered_change_source_segmentation_so_post_balance_not_original_old_group',
        dict(pre_balance_crossing_tokens=first['attention_valid_tokens_in_crossing_segments'],post_balance_crossing_tokens=second['attention_valid_tokens_in_crossing_segments']))

    qwen, worker = root/'qwen.py', root/'worker.py'
    qwen.write_text(T.QWEN_FIXTURE); worker.write_text(T.WORKER_FIXTURE)
    qwen.write_text(T.QWEN_FIXTURE.replace("'flash_attention_2':Qwen2FlashAttention2", "'sdpa':Qwen2FlashAttention2"))
    refused(lambda:P.native_qwen_caller(qwen,P.sha(qwen),worker,P.sha(worker)), 'changed_attention_class_mapping_refused')
    qwen.write_text(T.QWEN_FIXTURE.replace('return _flash_attention_forward','x = _flash_attention_forward(query_states,key_states,value_states,attention_mask,q_len,position_ids=position_ids)\n        return _flash_attention_forward'))
    refused(lambda:P.native_qwen_caller(qwen,P.sha(qwen),worker,P.sha(worker)), 'multiple_native_flash_calls_refused')
    qwen.write_text(T.QWEN_FIXTURE)
    refused(lambda:P.native_qwen_caller(qwen,'0'*64,worker,P.sha(worker)), 'native_qwen_byte_sha_mismatch_refused')
    refused(lambda:P.native_qwen_caller(qwen,P.sha(qwen),worker,'0'*64), 'actor_constructor_byte_sha_mismatch_refused')
    worker.write_text(T.WORKER_FIXTURE.replace('> 1','>= 1'))
    refused(lambda:P.native_qwen_caller(qwen,P.sha(qwen),worker,P.sha(worker)), 'sp1_monkeypatch_semantic_activation_refused')

    (root/'bundle').mkdir()
    run, package, package_sha, ready_sha = T.bundle(root/'bundle', A)
    verified = A.verify_bundle(run,package,package_sha,ready_sha,require_node=False)
    def inspect(): return P.inspect_bundle(A,verified,package,rules,0)
    out = inspect()
    assert len(out['inspections']) == 24 and out['actor_input_rows'] == 3840
    assert out['grouping_config']['rank_mini'] == 48 and out['grouping_config']['rank_mini_divisible_by_both_micros']
    assert not out['inspections'][-1]['inspection']['local_grouping']['4']['examples']
    record('eight_rank_full3840_zero_examples_and_mini48_cpu_positive')

    # Rehashing changed fixture config emulates an independently signed candidate;
    # protocol checks must reject it even if its byte identity is internally consistent.
    config_path = run/'resolved.yaml'; config_bytes = config_path.read_bytes()
    for field, value in [('ppo_mini_batch_size',376),('ppo_epochs',2)]:
        changed = json.loads(config_bytes); changed['actor_rollout_ref']['actor'][field] = value
        config_path.write_text(json.dumps(changed)); verified[3]['config_sha256'] = A.file_sha(config_path)
        refused(inspect, 'changed_'+field+'_refused_despite_rehashed_config')
    changed = json.loads(config_bytes); changed['data']['train_batch_size'] = 512
    config_path.write_text(json.dumps(changed)); verified[3]['config_sha256'] = A.file_sha(config_path)
    refused(inspect, 'changed_global_training_batch_refused_despite_rehashed_config')
    config_path.write_bytes(config_bytes); verified[3]['config_sha256'] = A.file_sha(config_path)

    # Byte/identity/shape binding failures use tiny fabricated .pt only.
    entry = verified[2]['workers'][0]['diagnostic_entries'][0]
    snapshot = Path(entry['snapshot_path']); original_bytes = snapshot.read_bytes()
    snapshot.write_bytes(original_bytes+b'x')
    refused(inspect, 'raw_snapshot_byte_tamper_refused'); snapshot.write_bytes(original_bytes)
    monitor = Path(verified[2]['workers'][0]['monitor_path']); monitor_bytes = monitor.read_bytes()
    monitor.write_bytes(monitor_bytes+b'x')
    refused(inspect, 'monitor_snapshot_byte_tamper_refused'); monitor.write_bytes(monitor_bytes)
    actor = run/'output/trajectories/outer-000111/actor-inputs.pt'
    actor_bytes = actor.read_bytes(); actor.write_bytes(actor_bytes+b'x')
    refused(inspect, 'full_actor_snapshot_byte_tamper_refused'); actor.write_bytes(actor_bytes)
    raw_json = Path(entry['path']); raw_bytes = raw_json.read_bytes()
    raw = json.loads(raw_bytes); raw['distribution'] = {'temperature':.5}
    raw_json.write_text(json.dumps(raw)); old_sha = entry['sha256']; entry['sha256'] = A.file_sha(raw_json)
    refused(inspect, 'raw_json_vs_snapshot_distribution_mismatch_refused')
    raw_json.write_bytes(raw_bytes); entry['sha256'] = old_sha
    worker_record = verified[2]['workers'][0]; worker_record['optimizer_committed'] = 1
    refused(inspect, 'worker_actual_optimizer_write_refused'); worker_record['optimizer_committed'] = 0
    worker_record['monitor_path'] = str(root/'outside-monitor.pt')
    refused(inspect, 'outside_run_monitor_path_refused'); worker_record['monitor_path'] = str(monitor)

    # Bind and inspect the real frozen source text without importing its modules.
    repo = WORK/'tree-safety110-a632ba9-v2/source/repo'
    trainer_path = repo/'verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(trainer_path.read_text())
    fit = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='fit')
    calls = [n for n in ast.walk(fit) if isinstance(n,ast.Call)]
    old = next(n for n in calls if ast.unparse(n.func)=='self.actor_rollout_wg.compute_log_prob' and
        len(n.args)==1 and ast.unparse(n.args[0])=='final_gen_batch_output')
    balance = next(n for n in calls if ast.unparse(n.func)=='self._balance_batch')
    save = next(n for n in calls if ast.unparse(n.func)=='save_rollouts')
    assert old.lineno < balance.lineno < save.lineno
    record('actual_frozen_source_old_logprob_before_balance_before_actor_save',
        dict(path=str(trainer_path),sha256=digest(trainer_path),old_call_line=old.lineno,balance_line=balance.lineno,actor_save_line=save.lineno))
    source_checks = [
        ('verl/protocol.py','DataProto','chunk','self.batch.chunk(chunks=chunks, dim=0)'),
        ('verl/workers/actor/dp_actor.py','DataParallelPPOActor','compute_log_prob','batch.split(micro_batch_size)'),
        ('verl/workers/actor/dp_actor.py','DataParallelPPOActor','update_policy','mini_batch.split(self.config.ppo_micro_batch_size)'),
    ]
    bound = []
    for name,cls_name,function_name,call_text in source_checks:
        path = repo/name; source_tree = ast.parse(path.read_text())
        cls = next(n for n in source_tree.body if isinstance(n,ast.ClassDef) and n.name==cls_name)
        function = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==function_name)
        assert any(isinstance(n,ast.Call) and ast.unparse(n)==call_text for n in ast.walk(function))
        bound.append(dict(path=str(path),sha256=digest(path),method=function_name,call=call_text))
    record('actual_frozen_source_contiguous_dp_and_fixed_micro_splits',bound)

assert digest(WORK/'inspect_tree_position_boundaries.py') == EXPECTED
assert digest(WORK/'test_inspect_tree_position_boundaries.py') == EXPECTED_TEST
assert not torch.cuda.is_initialized()
result = dict(status='INDEPENDENT_POSITION_SOURCE_CPU_PROBES_PASS',helper_sha256=EXPECTED,test_sha256=EXPECTED_TEST,
    checks=checks,checks_count=len(checks),actual_calibration_assets_loaded=False,actual_model_forwards=0,
    actual_optimizer_writes=0,remote_calls=0,cuda_initialized=False,scope='Source AST and small fabricated CPU fixtures; no runtime GPU attention, probability-cause attribution or training acceptance',
    cost=dict(wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu_started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)))
text = json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
(OUT/'INDEPENDENT_POSITION_CASES.json').write_text(text)
print(text)
