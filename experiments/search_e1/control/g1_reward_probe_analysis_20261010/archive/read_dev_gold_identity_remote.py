"""Read-only CPU compact question/source/alias identity; no answer rows emitted."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, json, re, unicodedata
import pyarrow.parquet as pq

BASE = Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
def hash_records(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
run = {'run_id':'f04fe593830c4831918ede872812c016'}
path = BASE / 'data/validation.parquet'; raw = path.read_bytes()
file_sha = hashlib.sha256(raw).hexdigest()
assert file_sha == '30d8a6175f3f3f7cc6d4e106521035be672cf9b35e98d371a8e81ac487859e71'
table = pq.read_table(path, columns=['prompt', 'data_source', 'reward_model'])
records = []
alias_counts = Counter()
for chat, source, reward in zip(table['prompt'].to_pylist(), table['data_source'].to_pylist(), table['reward_model'].to_pylist()):
    text = chat[0]['content']; marks = list(re.finditer(r'Question:\s*', text)); assert marks
    question = re.sub(r'\s+', ' ', text[marks[-1].end():]).strip().casefold()
    qid = hashlib.sha256(question.encode()).hexdigest()
    targets = reward['ground_truth']['target']
    if isinstance(targets, str):
        targets = [targets]
    aliases = sorted({re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(alias))).strip().casefold() for alias in targets})
    assert aliases and all(aliases)
    alias_counts[len(aliases)] += 1
    records.append([qid, source, aliases])
assert len(records) == len({r[0] for r in records}) == 512
print(json.dumps(dict(readonly=True, cpu_only=True, run_id=run['run_id'],
                      observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      validation_path=str(path), validation_file_sha256=file_sha, count=512,
                      question_id_normalization='last Question: suffix; whitespace collapse; strip; casefold; SHA256 UTF-8',
                      gold_alias_normalization='NFKC; whitespace collapse; strip; casefold; sorted unique aliases; metadata identity not EM normalization',
                      record_shape='[question_sha256,data_source,sorted_gold_aliases]',
                      hash_serialization='json.dumps(...,ensure_ascii=False,separators=(comma,colon)).encode UTF-8',
                      ordered_records_sha256=hash_records(records),
                      unordered_records_sha256=hash_records(sorted(records, key=lambda r:r[0])),
                      alias_counts=dict(alias_counts)), ensure_ascii=False))
