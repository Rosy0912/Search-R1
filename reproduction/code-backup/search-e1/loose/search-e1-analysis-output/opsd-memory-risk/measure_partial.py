import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from collections import Counter
os.nice(10)
base = Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007')
out = Path('/data/shaoshitong/search-e1-retriever-analysis-20261007/opsd-memory-risk')
out.mkdir(exist_ok=True)
sys.path[:0] = [str(base/'runtime/opsd/upstream/Search-R1'), str(base/'runtime/opsd')]
import torch
torch.set_num_threads(1)
from transformers import AutoTokenizer
from search_e1.core import mine_pairs
from search_e1.opsd import encoded_pair
assert not torch.cuda.is_initialized()

def rss():
    return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))) * 1024

def deep_size(value):
    seen = set()
    def visit(item):
        key = id(item)
        if key in seen:
            return 0
        seen.add(key)
        n = sys.getsizeof(item)
        if isinstance(item, dict):
            return n + sum(visit(k)+visit(v) for k,v in item.items())
        if isinstance(item, (list,tuple,set,frozenset)):
            return n + sum(visit(v) for v in item)
        return n
    return visit(value)

def summary(values):
    seq = sorted(values)
    if not seq:
        return {'count':0}
    return {'count':len(seq), 'mean':statistics.mean(seq), 'median':statistics.median(seq),
            'p95':seq[min(len(seq)-1,int(len(seq)*.95))], 'max':seq[-1]}

proof = {'time':time.time(), 'pid':os.getpid(),
         'scope':'Offline bounded statistics only. Partial trajectories are never used for training or written as a training pair file.',
         'max_context':8192, 'questions_per_shard':256, 'sampling':'First 256 complete question groups in each current shard, preserving every 5 siblings and source order. This is not a randomized full-dataset sample.',
         'runtime_sha256':{r:hashlib.sha256((base/'runtime/opsd'/r).read_bytes()).hexdigest() for r in ['search_e1/core.py','search_e1/mine.py','search_e1/opsd.py','search_e1/io.py']},
         'rss_imports_bytes':rss(), 'shards':[]}
rows=[]
for path in sorted((base/'runs/rollouts.jsonl.parts').glob('rollouts-*.jsonl')):
    selected=[];prefix=[]
    with path.open() as f:
        for i,line in enumerate(f):
            if i == 256*5:
                break
            selected.append(json.loads(line));prefix.append(line)
    groups=Counter(r['question_id'] for r in selected)
    assert len(groups)==256 and set(groups.values())=={5}
    rows.extend(selected)
    proof['shards'].append({'path':str(path),'questions':len(groups),'rows':len(selected),
                            'prefix_sha256':hashlib.sha256(''.join(prefix).encode()).hexdigest()})
    del selected,prefix
proof['rss_rows_bytes']=rss()
proof['raw_rows_deep_bytes']=deep_size(rows)
proof['source_rows']=len(rows)
proof['source_questions']=len({r['question_id'] for r in rows})
proof['source_datasets']=dict(Counter(r.get('data_source','unknown') for r in rows))
proof['source_row_bytes']=summary([len(json.dumps(r,ensure_ascii=False).encode())+1 for r in rows])
dataset_by_question={r['question_id']:r.get('data_source','unknown') for r in rows}
raw_mined = mine_pairs(rows)
proof['mined_pairs']=len(raw_mined)
proof['mined_datasets']=dict(Counter(dataset_by_question[p['question_id']] for p in raw_mined))
# Independent JSON loads reproduce each rank's original read_jsonl object shape.
pairs=[json.loads(json.dumps(p,ensure_ascii=False)) for p in raw_mined]
proof['raw_pairs_deep_bytes']=deep_size(pairs)
proof['pair_json_bytes']=summary([len(json.dumps(p,ensure_ascii=False).encode())+1 for p in pairs])
del raw_mined
gc.collect()
proof['rss_before_tokenizer_bytes']=rss()
tokenizer=AutoTokenizer.from_pretrained(base/'assets/archive100',use_fast=True)
proof['rss_before_encoding_bytes']=rss()
encoded=[encoded_pair(tokenizer,p,8192) for p in pairs]
proof['rss_after_encoding_bytes']=rss()
examples=[item for item in encoded if item is not None]
proof['usable_pairs']=len(examples)
proof['skipped_pairs']=len(encoded)-len(examples)
proof['encoded_deep_bytes']=deep_size(encoded)
proof['encoded_and_examples_deep_bytes']=deep_size([encoded,examples])
proof['encoded_item_deep_bytes']=summary([deep_size(e) for e in examples])
proof['student_prefix_tokens']=summary([len(e[0][0]) for e in examples])
proof['teacher_prefix_tokens']=summary([len(e[0][1]) for e in examples])
proof['response_tokens']=summary([len(e[1]) for e in examples])
proof['valid_policy_tokens']=summary([sum(e[2]) for e in examples])
proof['total_id_tokens']=summary([len(e[0][0])+len(e[0][1])+len(e[1]) for e in examples])
counts=Counter();max_question_bytes=0;max_question=None
with (base/'data/train.jsonl').open() as f:
    for line in f:
        record=json.loads(line);counts[record.get('data_source','unknown')]+=1
        n=len(record['question'].encode())
        if n>max_question_bytes:max_question_bytes=n;max_question=record['question_id']
proof['full_train_dataset_counts']=dict(counts)
proof['full_train_questions']=sum(counts.values())
proof['full_train_rollouts']=sum(counts.values())*5
proof['full_train_max_question_utf8_bytes']=max_question_bytes
proof['full_train_max_question_id']=max_question
rate=len(pairs)/proof['source_questions']
predicted_pairs=rate*proof['full_train_questions']
raw_per_pair=proof['raw_pairs_deep_bytes']/len(pairs)
enc_per_pair=proof['encoded_and_examples_deep_bytes']/len(pairs)
rss_per_pair=max(0,proof['rss_after_encoding_bytes']-proof['rss_before_encoding_bytes'])/len(pairs)
proof['projection_based_only_on_current_sample']={
    'mine_rate':rate,'projected_pairs':predicted_pairs,
    'raw_pairs_deep_bytes_per_pair':raw_per_pair,
    'encoded_deep_bytes_per_input_pair':enc_per_pair,
    'observed_encoding_rss_bytes_per_input_pair':rss_per_pair,
    'eight_rank_raw_plus_encoded_deep_bytes':8*predicted_pairs*(raw_per_pair+enc_per_pair),
    'eight_rank_all_questions_have_pair_deep_bytes':8*proof['full_train_questions']*(raw_per_pair+enc_per_pair),
    'uncertainty':'Current sample represents only datasets present in source_datasets. Unobserved Hotpot length/mine/filter distributions can differ. Python deep size excludes allocator padding/fragmentation and process/model/library baseline.'}
proof['cuda_initialized']=torch.cuda.is_initialized()
assert not proof['cuda_initialized']
proof['complete_time']=time.time()
path=out/'partial-statistics.json';path.write_text(json.dumps(proof,indent=2))
print(json.dumps(proof,indent=2),flush=True)
