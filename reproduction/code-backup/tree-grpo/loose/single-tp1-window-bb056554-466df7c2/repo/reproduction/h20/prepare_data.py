"""Use the pinned upstream prompt function and original NQ fields, offline."""
import json
from pathlib import Path
import runpy
import sys
from datasets import Dataset

base = Path('/data/search-r1')
sys.path.insert(0, str(base/'repo'))
prefix = runpy.run_path(str(base/'repo/scripts/data_process/nq_search.py'))['make_prefix']
out = base/'data/nq'
out.mkdir(parents=True, exist_ok=True)
for split in ['train', 'test']:
    rows = []
    with (base/'assets/nq'/f'{split}.jsonl').open() as f:
        for idx, line in enumerate(f):
            example = json.loads(line)
            example['question'] = example['question'].strip()
            if not example['question'].endswith('?'):
                example['question'] += '?'
            rows.append(dict(data_source='nq',
                prompt=[dict(role='user',content=prefix(example, 'base'))],
                ability='fact-reasoning',
                reward_model=dict(style='rule',ground_truth=dict(target=example['golden_answers'])),
                extra_info=dict(split=split,index=idx)))
    Dataset.from_list(rows).to_parquet(str(out/f'{split}.parquet'))
    print('NQ_PARQUET', split, len(rows), flush=True)
