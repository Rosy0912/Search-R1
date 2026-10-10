"""Read-only CPU G1 dev512 cohort/protocol provenance, no generated outputs."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, json, re, socket
import pyarrow.parquet as pq

BASE = Path('/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512')
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
run = json.loads((BASE / 'logs/run-manifest.json').read_text())
mp = BASE / 'data/manifest.json'; mb = mp.read_bytes(); manifest = json.loads(mb)
record = manifest['splits']['validation']; path = BASE / 'data' / record['path']
raw = path.read_bytes(); assert sha(raw) == record['sha256'] == run['input_sha256']['validation.parquet']
table = pq.read_table(path, columns=['prompt', 'data_source'])
ids = []
for chat in table['prompt'].to_pylist():
    text = chat[0]['content']; matches = list(re.finditer(r'Question:\s*', text)); assert matches
    question = re.sub(r'\s+', ' ', text[matches[-1].end():]).strip().casefold()
    ids.append(sha(question.encode()))
assert len(ids) == len(set(ids)) == 512
cp = BASE / 'logs/resolved-training-config.json'; cb = cp.read_bytes(); cfg = json.loads(cb)['resolved_config']
out = dict(readonly=True, cpu_only=True, run_id=run['run_id'], node=socket.gethostname(),
           observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           manifest_path=str(mp), manifest_sha256=sha(mb), validation_path=str(path), validation_sha256=sha(raw),
           question_count=512, source_counts=dict(Counter(table['data_source'].to_pylist())),
           normalized_question_identity_definition='last Question: suffix; whitespace collapse; strip; casefold; UTF-8 SHA256',
           question_order_compact_sha256=sha(json.dumps(ids, separators=(',', ':')).encode()),
           question_set_compact_sha256=sha(json.dumps(sorted(ids), separators=(',', ':')).encode()),
           resolved_config_path=str(cp), resolved_config_file_sha256=sha(cb),
           generation_configured=dict(max_turns=cfg['max_turns'], do_search=cfg['do_search'],
                                      data={k:cfg['data'][k] for k in ('max_start_length','max_prompt_length','max_response_length','max_obs_length','val_batch_size')},
                                      rollout=cfg['actor_rollout_ref']['rollout'], retriever=cfg['retriever']),
           source=[])
for relative in ('verl/trainer/ppo/ray_trainer.py', 'search_r1/llm_agent/generation.py',
                 'verl/workers/rollout/vllm_rollout/vllm_rollout.py', 'reproduction/common8/validation_identity.py'):
    p = BASE / 'repo' / relative
    if not p.is_file():
        continue
    b = p.read_bytes(); assert sha(b) == run['source_sha256'][relative]
    lines = b.decode().splitlines(); excerpts = []
    for i, line in enumerate(lines):
        if any(term in line for term in ("'do_sample': False", "'validate': True", 'sampling_params', 'max_turns', 'temperature=0', 'temperature = 0', 'drop_last=False', "'validation' if step", 'max_obs_length', 'is_validation')):
            excerpts.append(dict(line=i+1, source='\n'.join(lines[max(0,i-1):min(len(lines),i+3)])))
    out['source'].append(dict(path=str(p), sha256=sha(b), bound_sha256_match=True, excerpts=excerpts))
print(json.dumps(out))
