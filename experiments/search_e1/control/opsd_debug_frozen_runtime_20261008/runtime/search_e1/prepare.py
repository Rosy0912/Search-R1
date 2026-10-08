"""Create held-out training validation and untouched seven-benchmark evaluation splits."""
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
import os
from .core import prompt
from .io import write_jsonl, write_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data')
    p.add_argument('--validation-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    train, validation, test = [], [], []
    counts = {}
    for source in ['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']:
        files = {p.stem: str(p) for p in (Path(os.environ['E1_DATASET_ROOT']) / source).glob('*.jsonl')}
        if not files:
            raise ValueError('No pinned dataset files for ' + source)
        ds = load_dataset('json', data_files=files)
        def convert(split):
            return [{'question_id': f'{source}:{split}:{i}', 'question': r['question'].strip(),
                     'answers': r['golden_answers'], 'data_source': source} for i, r in enumerate(ds[split])]
        if source in ['nq', 'hotpotqa']:
            import random
            rows = convert('train')
            random.Random(a.seed).shuffle(rows)
            n = a.validation_size // 2
            validation.extend(rows[:n])
            train.extend(rows[n:])
        split = 'test' if source in ['nq','triviaqa','popqa','bamboogle'] else 'dev'
        rows = convert(split)
        test.extend(rows)
        counts[source] = {'split': split, 'count': len(rows)}
    for name, rows in [('train', train), ('validation', validation), ('test', test)]:
        write_jsonl(out/f'{name}.jsonl', rows)
        # Include an example answer in GRPO instruction to accommodate upstream EM extractor.
        records = [{'data_source': r['data_source'],
                    'prompt': [{'role':'user','content':prompt(r['question'])[0]['content']}],
                    'ability':'fact-reasoning', 'reward_model':{'style':'rule','ground_truth':{'target':r['answers']}},
                    'extra_info':{'index':r['question_id'],'split':name}} for r in rows]
        Dataset.from_list(records).to_parquet(str(out/f'{name}.parquet'))
    write_json(out/'manifest.json', {'train':len(train),'validation':len(validation),'test':counts,'seed':a.seed,
                                    'validation_policy':'Held out from NQ/Hotpot training; never select on benchmark test/dev.'})
if __name__ == '__main__':
    main()
