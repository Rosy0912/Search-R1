"""CPU contract tests; fake engine is never hardware/model quality evidence."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess
import pytest
import torch
from safetensors.torch import save_file, load_file

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ['AGENT_ROOT']) / 'repo' if os.environ.get('AGENT_ROOT') else HERE.parent / 'topd'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import eval64 as e
import supervisor as s
import runtime_safety as safety
from verl import DataProto


class Tokenizer:
    pad_token_id = eos_token_id = 0
    bos_token_id = None
    pad_token = ''
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(int(v)) for v in ids if int(v))
    def batch_decode(self, rows, **kwargs):
        return [self.decode(row) for row in rows]
    def __call__(self, texts, **kwargs):
        ids = [self.encode(t) for t in texts]
        width = max(map(len, ids), default=0)
        return {'input_ids': torch.tensor([r + [0] * (width-len(r)) for r in ids], dtype=torch.long)}


def prompts(count=2):
    idx = torch.tensor([[0, 12, 13]] * count)
    return DataProto.from_dict(dict(input_ids=idx, attention_mask=torch.tensor([[0, 1, 1]] * count),
                                    position_ids=torch.tensor([[0, 0, 1]] * count)),
                               meta_info=dict(do_sample=False, eos_token_id=0, pad_token_id=0))


class Engine:
    def __init__(self, raw=None):
        self.raw, self.calls = raw or [[91, 0]], []
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(prompt_token_ids=ids, outputs=[SimpleNamespace(token_ids=self.raw[i % len(self.raw)],
                    finish_reason='stop', stop_reason=None)]) for i, ids in enumerate(kwargs['prompt_token_ids'])]


def test_adapter_original_padding_eos_positions_and_raw_capture(tmp_path):
    engine = Engine([[91, 92, 0], [94, 0]])
    sampling = object()
    adapter = e.StandaloneAdapter(engine, Tokenizer(), sampling, tmp_path)
    result = adapter.generate_sequences(prompts())
    assert engine.calls[0]['sampling_params'] is sampling
    assert engine.calls[0]['prompt_token_ids'] == [[12, 13], [12, 13]]
    assert result.batch['responses'].shape == (2, 500)
    assert result.batch['attention_mask'][0, :8].tolist() == [0, 1, 1, 1, 1, 1, 0, 0]
    assert result.batch['position_ids'][0, 3:6].tolist() == [2, 3, 4]
    artifact = load_file(str(tmp_path / 'generation-01.safetensors'))
    assert torch.equal(artifact['responses'], result.batch['responses'])
    assert json.loads((tmp_path / 'generation-01.json').read_text())['raw_response_token_ids'] == [[91, 92, 0], [94, 0]]


def test_sampling_rejected_before_engine(tmp_path):
    engine = Engine()
    batch = prompts()
    batch.meta_info['do_sample'] = True
    with pytest.raises(RuntimeError, match='greedy'):
        e.StandaloneAdapter(engine, Tokenizer(), object(), tmp_path).generate_sequences(batch)
    assert not engine.calls


@pytest.mark.parametrize('fault', ['order', 'n', 'length', 'count'])
def test_bad_engine_contract_rejected(tmp_path, fault):
    class Bad(Engine):
        def generate(self, **kwargs):
            out = super().generate(**kwargs)
            if fault == 'order': out[0].prompt_token_ids = [99]
            if fault == 'n': out[0].outputs *= 2
            if fault == 'length': out[0].outputs[0].token_ids = [91] * 501
            if fault == 'count': out.pop()
            return out
    with pytest.raises(RuntimeError):
        e.StandaloneAdapter(Bad(), Tokenizer(), object(), tmp_path).generate_sequences(prompts())


@pytest.mark.parametrize('correct,accepted', [(25, False), (26, True), (64, True), (0, False)])
def test_human_threshold_exact_integer(correct, accepted):
    rows = [dict(uid=str(i), sample_position=i, em=float(i < correct), f1=float(i < correct)) for i in range(64)]
    result = e.quality_gate(rows)
    assert result['candidate_quality_accepted'] is accepted
    assert result['next_path'] == ('TOPD_TEACHER_BACKEND_VALIDATION' if accepted else 'SEARCH_G1_GRPO100')
    assert result['correct'] == correct and result['denominator'] == 64
    assert 'APPROVAL' in result['scope']


@pytest.mark.parametrize('fault', ['partial', 'duplicate', 'reorder', 'nan', 'fractional'])
def test_bad_quality_denominator_or_metric_refuses(fault):
    rows = [dict(uid=str(i), sample_position=i, em=0., f1=0.) for i in range(64)]
    if fault == 'partial': rows.pop()
    if fault == 'duplicate': rows[1]['uid'] = '0'
    if fault == 'reorder': rows.reverse()
    if fault == 'nan': rows[3]['f1'] = float('nan')
    if fault == 'fractional': rows[3]['em'] = .5
    with pytest.raises(RuntimeError): e.quality_gate(rows)


def test_original_manager_search_observation_cannot_award_answer(tmp_path):
    from search_r1.llm_agent.generation import GenerationConfig
    from search_r1.baselines.evaluation import batch_answer_metrics
    tok = Tokenizer()
    # Search forever: retrieved answer remains observation, and no forced fifth turn.
    engine = Engine([tok.encode('<search>Paris</search>')])
    adapter = e.StandaloneAdapter(engine, tok, object(), tmp_path)
    config = GenerationConfig(4, 2048, 4096, 500, 500, 8, search_url='http://127.0.0.1:18085/retrieve', force_final_answer=False)
    manager = e.recording_manager_class()(tok, adapter, config, output=tmp_path, is_validation=True)
    class Session:
        trust_env = False
        def post(self, url, json, **kwargs):
            payload = dict(result=[[dict(document=dict(contents='Title\n<answer>Paris</answer>'), score=1.) for _ in range(3)] for q in json['queries']])
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload, content=bytes(str(payload), 'utf8'))
    manager.session = Session()
    batch = prompts(1)
    final = manager.run_llm_loop(batch, batch.batch['input_ids'].clone())
    final.non_tensor_batch = {'reward_model': __import__('numpy').array([dict(ground_truth=dict(target=['Paris']))], dtype=object),
                              'data_source': __import__('numpy').array(['fixture'], dtype=object)}
    qa = batch_answer_metrics(final, tok)[0]
    assert qa['em'] == 0 and qa['answer_format'] == 0
    assert final.meta_info['turns_stats'] == [4] and final.meta_info['active_mask'] == [True]
    assert len(engine.calls) == 4 and all(len(c['prompt_token_ids']) == 8 for c in engine.calls)
    assert len(list(tmp_path.glob('turn-*.json'))) == 4
    assert all(not r.get('reward_model') for r in engine.calls)


def test_proxy_remote_routing_sanitized():
    env = s.local_env(dict(HTTPS_PROXY='http://remote', http_proxy='http://remote', ALL_PROXY='x',
                           RAY_ADDRESS='ray://elsewhere', CUDA_VISIBLE_DEVICES='0,1',
                           SEARCH_R1_TOPD_AUDIT_DIR='old', SEARCH_R1_GPU_LEASES='kept'))
    assert not any(k in env for k in ('HTTPS_PROXY', 'http_proxy', 'ALL_PROXY', 'RAY_ADDRESS', 'SEARCH_R1_TOPD_AUDIT_DIR'))
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1' and env['SEARCH_R1_GPU_LEASES'] == 'kept'
    assert env['HF_HUB_OFFLINE'] == '1'


def test_no_slurm_refuses_before_assets_or_gpu(monkeypatch):
    monkeypatch.setenv('AGENT_ROOT', str(s.ROOT))
    monkeypatch.setattr(s.socket, 'gethostname', lambda: 'h100-11')
    monkeypatch.setenv('SLURM_JOB_NODELIST', 'h100-11')
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'): s.slurm_gate()


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    root = tmp_path / 'root'; model = root / 'assets/32b'; model.mkdir(parents=True)
    data = root / 'validation.parquet'; data.write_bytes(b'fixture64')
    monkeypatch.setattr(e, 'DATA_SHA', e.sha(data))
    weight_map = {}
    for i in range(17):
        name = 'model-%05d-of-00017.safetensors' % (i+1)
        key = 'weight%d' % i
        save_file({key: torch.ones(2, dtype=torch.bfloat16)}, str(model / name))
        weight_map[key] = name
    (model / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=weight_map)))
    config = dict(model_type='qwen2', architectures=['Qwen2ForCausalLM'], num_hidden_layers=64,
                  hidden_size=5120, num_attention_heads=40, num_key_value_heads=8, eos_token_id=151643, torch_dtype='bfloat16')
    (model / 'config.json').write_text(json.dumps(config))
    for name in ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'vocab.json', 'merges.txt'): (model / name).write_text('{}')
    repo = root / 'repo'; repo.mkdir()
    sources = {}
    for name in e.REUSED_SOURCES:
        path = repo / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
        sources[name] = e.sha(path)
    (root / 'tools').mkdir(); (root / 'tools/topd-short-run.py').write_text('fixture-owned')
    metadata = dict(id='Qwen/Qwen2.5-32B', sha=e.REVISION, siblings=[])
    for file in model.iterdir():
        raw = file.read_bytes()
        record = dict(rfilename=file.name, size=len(raw), blobId=e.hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest())
        if file.suffix == '.safetensors': record['lfs'] = dict(sha256=e.sha(file))
        metadata['siblings'].append(record)
    metadata_path = root / 'official-metadata.json'; metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(e, 'OFFICIAL_METADATA_SHA', e.sha(metadata_path))
    runtime = {}
    for name in ('assets-root/assets/wiki/e5_Flat.index', 'assets-root/assets/wiki/wiki-18.jsonl',
                 'assets-root/assets/e5-base-v2/model.safetensors', 'retriever-gpu-lib/faiss/lib.so'):
        file = root / name; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(b'cpu-fixture-not-runtime-asset')
        runtime[name] = e.sha(file)
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: 'fixture-head\n')
    binding = dict(status='INVENTORY_VERIFIED', revision=e.REVISION, model_directory=str(model),
                   data_path=str(data), data_sha256=e.sha(data), code_commit='fixture-head',
                   source_files=sources, tool_files={n: e.sha(HERE / n) for n in ('eval64.py', 'supervisor.py', 'runtime_safety.py', 'eval64.sbatch')},
                   official_metadata_path=str(metadata_path), official_metadata_sha256=e.sha(metadata_path), runtime_files=runtime,
                   owned_launcher_sha256=e.sha(root / 'tools/topd-short-run.py'),
                   model_files={p.name: e.sha(p) for p in model.iterdir()})
    path = root / 'binding.json'; path.write_text(json.dumps(binding))
    return root, model, path, binding


def test_complete_actual_tiny_inventory_pass_cpu_only(inventory):
    root, model, path, binding = inventory
    assert e.asset_gate(path, root)[1] == model


@pytest.mark.parametrize('fault', ['partial', 'filedrift', 'instruct', 'dtype', 'index', 'tool', 'source', 'missingbinding'])
def test_inventory_fails_closed_before_gpu(inventory, fault):
    root, model, path, binding = inventory
    if fault == 'partial': binding['status'] = 'DOWNLOADING'
    if fault == 'filedrift': (model / 'tokenizer.json').write_text('tampered')
    if fault == 'instruct': binding['revision'] = 'anothermodel'
    if fault == 'dtype':
        name = next(n for n in binding['model_files'] if n.endswith('.safetensors'))
        i = int(name.split('-')[1])-1
        save_file({'weight%d' % i: torch.ones(2)}, str(model / name))
        binding['model_files'][name] = e.sha(model / name)
    if fault == 'index':
        index_path = model / 'model.safetensors.index.json'; idx = json.loads(index_path.read_text())
        idx['weight_map']['unavailable'] = next(iter(idx['weight_map'].values()))
        index_path.write_text(json.dumps(idx)); binding['model_files'][index_path.name] = e.sha(index_path)
    if fault == 'tool': binding['tool_files']['eval64.py'] = '0' * 64
    if fault == 'source': binding['source_files']['search_r1/llm_agent/generation.py'] = '0' * 64
    if fault == 'missingbinding': binding['model_files'].pop('tokenizer.json')
    path.write_text(json.dumps(binding))
    with pytest.raises(RuntimeError): e.asset_gate(path, root)


def test_exclusive_json_refuses_overwrite(tmp_path):
    path = tmp_path / 'record.json'; e.write_json(path, {'first': 1})
    with pytest.raises(FileExistsError): e.write_json(path, {'second': 2})
    assert json.loads(path.read_text()) == {'first': 1}


def test_direct_eval_cannot_bypass_supervisor(monkeypatch):
    monkeypatch.delenv('SEARCH_R1_EVAL64_SUPERVISOR_PID', raising=False)
    with pytest.raises(RuntimeError): e.child_execution_gate('/tmp')


def dataset_tokenizer():
    from tokenizers import Tokenizer as Backend
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    backend = Backend(WordLevel({'[UNK]': 0, '[PAD]': 1, 'user': 2, 'assistant': 3, 'question': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]')
    tokenizer.chat_template = "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    return tokenizer


def test_real_parquet_original_dataset_sample42_and_single_chat_template(tmp_path):
    import pandas as pd
    path = tmp_path / '64.parquet'
    frame = pd.DataFrame([dict(prompt=[dict(role='user', content='question')], data_source='fixture',
                              reward_model=dict(ground_truth=dict(target=['answer'])), extra_info=dict(index=i)) for i in range(64)])
    frame.to_parquet(path)
    tokenizer = dataset_tokenizer()
    dataset, ids = e.load_questions(path, tokenizer, tmp_path)
    expected = frame.sample(64, random_state=42)['extra_info'].map(lambda r: r['index']).tolist()
    assert [r['original_row']['extra_info']['index'] for r in ids] == expected
    row = dataset[0]
    assert tokenizer.decode(row['input_ids'][row['attention_mask'].bool()]) == 'user question assistant'
    assert int(row['attention_mask'].sum()) == 3


def test_actual_dataset_overlength_errors_not_silently_filtered_or_truncated(tmp_path):
    import pandas as pd
    path = tmp_path / 'long.parquet'
    pd.DataFrame([dict(prompt=[dict(role='user', content='question ' * 4096)], data_source='fixture',
                       extra_info=dict(index=i)) for i in range(64)]).to_parquet(path)
    dataset, _ = e.load_questions(path, dataset_tokenizer(), tmp_path)
    assert len(dataset) == 64
    with pytest.raises(NotImplementedError): dataset[0]


def active_fixture():
    uids = {'GPU-%d' % i for i in range(8)}
    gpu = [dict(index=str(i), uuid='GPU-%d' % i, used_mib=8000., total_mib=81920., sm=0.) for i in range(8)]
    old = [dict(uuid='GPU-%d' % i, pid=100+i, used_mib=1100.) for i in range(8)]
    identities = {str(100+i): dict(pid=100+i, starttime_ticks=200+i, cgroup=['0::/outside-baseline']) for i in range(8)}
    baseline = dict(gpus=gpu, applications=old, identities=identities)
    service = dict(pid=999, starttime_ticks=1000, cgroup=['0::/slurm/job_42/step_batch'])
    snapshot = dict(gpus=[dict(g) for g in gpu], applications=[dict(p) for p in old]+[dict(uuid=uid, pid=999, used_mib=6900.) for uid in uids],
                    identities=dict(identities, **{'999': service}))
    sm = {p['uuid']: 0. for p in old}
    return snapshot, baseline, uids, service, '/slurm/job_42/step_batch', sm


def test_actual_sized_retriever_passes_original_absolute15_and_accounting():
    snapshot, base, uids, service, group, sm = active_fixture()
    result = safety.accounting_gate(snapshot, base, uids, service, group, sm, pre_model=True)
    assert len(result['own_pid_uuids']) == 8 and result['gpu_memory_utilization'] == .4
    assert all(g['used_mib'] / g['total_mib'] <= .15 for g in snapshot['gpus'])


def test_owned_model_active_can_exceed15_but_before_model_cannot():
    snapshot, base, uids, service, group, sm = active_fixture()
    for g in snapshot['gpus']: g['used_mib'] = 32000.
    safety.accounting_gate(snapshot, base, uids, service, group, sm, pre_model=False)
    with pytest.raises(RuntimeError, match='absolute'): safety.accounting_gate(snapshot, base, uids, service, group, sm, pre_model=True)


@pytest.mark.parametrize('fault', ['unknown', 'foreignmemory', 'foreignsm', 'pidreuse', 'scope', 'serviceuuid', 'unleased', 'free', 'unknownsm'])
def test_active_accounting_rejects_unknown_or_foreign_change(fault):
    snapshot, base, uids, service, group, sm = active_fixture()
    if fault == 'unknown':
        snapshot['applications'].append(dict(uuid='GPU-0', pid=1234, used_mib=500.))
        snapshot['identities']['1234'] = dict(pid=1234, starttime_ticks=1, cgroup=['0::/outside'])
    if fault == 'foreignmemory': snapshot['applications'][0]['used_mib'] = 15000.
    if fault == 'foreignsm': sm['GPU-0'] = 16.
    if fault == 'pidreuse': snapshot['identities']['100'] = dict(snapshot['identities']['100'], starttime_ticks=999)
    if fault == 'scope': snapshot['identities']['999'] = dict(service, cgroup=['0::/outside'])
    if fault == 'serviceuuid': snapshot['applications'] = [p for p in snapshot['applications'] if not(p['pid'] == 999 and p['uuid'] == 'GPU-0')]
    if fault == 'unleased': snapshot['applications'].append(dict(uuid='GPU-outsider', pid=999, used_mib=1.))
    if fault == 'free': snapshot['gpus'][0]['used_mib'] = 50000.
    if fault == 'unknownsm': sm['GPU-0'] = float('nan')
    with pytest.raises(RuntimeError): safety.accounting_gate(snapshot, base, uids, service, group, sm, pre_model=True)


def test_small_foreign_memory_change_within15_does_not_invent_zero_delta_guard():
    snapshot, base, uids, service, group, sm = active_fixture()
    snapshot['applications'][0]['used_mib'] += 1.
    safety.accounting_gate(snapshot, base, uids, service, group, sm, pre_model=True)



# Actual GPU3009 verifies this backend; these CPU cases remain contracts.
F, O = {('u',1)}, {('u',2)}
def run(sm=80, status='SUCCESS', rows=None, start=1000000, end=2200000, now=2200000):
    rows = [dict(pid=2,timestamp_us=2000000,sm=80)] if rows is None else rows
    return safety.sm_aggregate_bounds({'gpus':[dict(uuid='u',sm=sm)]},F,O,
        {'u':dict(status=status,start_us=start,end_us=end,samples=rows)},now)

def test_idle_notfound_device_bound_is_not_pid_zero():
    bound, ev = run(sm=7,status='NVML_ERROR_6',rows=[])
    assert bound == {'u':7}
    assert ev['u']['absent_pid_values']=='NOT_INFERRED'

def test_active_supported_missing_foreign_infers_only_nonzero_filter():
    bound, ev=run()
    assert bound=={'u':0}
    assert ev['u']['absent_interpretation']=='INFERRED_ZERO_NONZERO_FILTER_NOT_MEASURED'
    assert ev['u']['positive_own_pids']==[2]

def test_supported_foreign_nonzero_is_counted():
    bound,ev=run(rows=[dict(pid=2,timestamp_us=2000000,sm=80),dict(pid=1,timestamp_us=2000000,sm=16)])
    assert bound['u']==16
    assert ev['u']['foreign_observed_maxima']=={1:16}

@pytest.mark.parametrize('status',['NVML_ERROR_6','NVML_ERROR_3','UNKNOWN'])
def test_active_errors_never_infer_zero(status):
    with pytest.raises(Exception): run(status=status)

@pytest.mark.parametrize('rows',[
    [],[dict(pid=2,timestamp_us=999999,sm=80)],
    [dict(pid=2,timestamp_us=2200001,sm=80)],
    [dict(pid=2,timestamp_us=2000000,sm=0)],
    [dict(pid=3,timestamp_us=2000000,sm=80)],
    [dict(pid=2,timestamp_us=2000000,sm=float('nan'))],
    [dict(pid=2,timestamp_us=2000000,sm=80),dict(pid=1,timestamp_us=2100000,sm=1)]])
def test_invalid_positive_or_incomplete_period_refuses(rows):
    with pytest.raises(Exception): run(rows=rows)

def test_stale_query_refuses():
    with pytest.raises(Exception): run(now=5000000)

def test_cross_gpu_own_positive_cannot_prove_support():
    with pytest.raises(Exception):
        safety.sm_aggregate_bounds({'gpus':[dict(uuid='other',sm=80)]},{('other',1)},O,
          {'other':dict(status='SUCCESS',start_us=1000000,end_us=2200000,
                       samples=[dict(pid=2,timestamp_us=2000000,sm=80)])},2200000)


def test_window_end_fresh_but_sample_old_refuses():
    with pytest.raises(Exception): run(now=4100000)


@pytest.mark.parametrize('initial_device_sm,final_device_sm,expected_bound',[(90,0,0),(7,2,7)])
def test_active_high_then_idle_does_not_charge_own_activity_to_foreign(monkeypatch,initial_device_sm,final_device_sm,expected_bound):
    import runtime_safety
    import search_r1.baselines.topd_local_gpu as backend
    import time
    now=int(time.time()*1000000)
    gpu=[dict(index=str(i),uuid='u'+str(i),used_mib=8000,total_mib=80000,sm=initial_device_sm) for i in range(8)]
    ids={str(i+1):dict(pid=i+1,starttime_ticks=1,cgroup=['0::/foreign']) for i in range(8)}
    ids['99']=dict(pid=99,starttime_ticks=1,cgroup=['0::/ours'])
    apps=[dict(uuid='u'+str(i),pid=i+1,used_mib=1000) for i in range(8)]
    base=dict(gpus=gpu,applications=apps,identities=ids)
    first=dict(gpus=gpu,applications=apps+[dict(uuid='u'+str(i),pid=99,used_mib=7000) for i in range(8)],identities=ids)
    final=dict(first,gpus=[dict(g,sm=final_device_sm) for g in gpu])
    seq=iter([first,final]);monkeypatch.setattr(safety,'nvml_snapshot',lambda:next(seq))
    monkeypatch.setattr(safety.subprocess,'check_output',lambda *a,**kw:'original pmon -')
    monkeypatch.setattr(backend,'lease_ownership',lambda r:None)
    monkeypatch.setenv('SEARCH_R1_GPU_PREFLIGHT','1')
    monkeypatch.setenv('SEARCH_R1_GPU_LEASES',safety.json.dumps([dict(uuid='u'+str(i)) for i in range(8)]))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(map(str,range(8))))
    windows={'u'+str(i):dict(status='SUCCESS',start_us=now-1200000,end_us=now,
                  samples=[dict(pid=99,timestamp_us=now-200000,sm=90)]) for i in range(8)}
    monkeypatch.setattr(safety,'collect_process_windows',lambda *a:(windows,now))
    result=safety.active_gate(dict(baseline=base,retriever_identity=ids['99'],cgroup='/ours'))
    assert set(result['foreign_sm_aggregate_upper_bounds'].values())=={expected_bound}
    assert result['foreign_sm_evidence']['u0']['initial_foreign_upper_bound']==expected_bound
