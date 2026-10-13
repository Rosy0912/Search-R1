"""CPU identity, immutable score ledger, and complete final evaluation acceptance."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

FIELDS = ('source', 'f1', 'em', 'answer_format', 'format_rate', 'generated_tokens', 'search_calls')
EXPECTED_COUNTS = dict(nq=3610, triviaqa=11313, popqa=14267, hotpotqa=7405,
                       **{'2wikimultihopqa':12576}, musique=2417, bamboogle=125)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def files(config):
    value = config.data.final_val_files
    return [str(value)] if isinstance(value, str) else [str(x) for x in value]

def asset_manifest(configpath, modelpath):
    """Hashes bytes, never loads tensors or allocates GPU memory."""
    from omegaconf import OmegaConf
    config = OmegaConf.load(configpath)
    OmegaConf.resolve(config)
    data = [dict(path=str(Path(p).resolve()), sha256=digest(p)) for p in files(config)]
    model = Path(modelpath).resolve()
    if not model.is_dir():
        raise ValueError('Missing explicit HF checkpoint')
    assets = [dict(name=str(p.relative_to(model)), sha256=digest(p))
              for p in sorted(model.rglob('*')) if p.is_file()]
    if not any(x['name'].endswith(('.safetensors', '.bin')) for x in assets):
        raise ValueError('HF model has no weights')
    return dict(config_sha256=digest(configpath), dataset_sha256=object_sha(data),
                model_sha256=object_sha(assets), dataset_files=data, model_files=assets,
                dataset_count=51713, prompt_key=str(config.data.prompt_key), expected_source_counts=EXPECTED_COUNTS)

def atomic_immutable(path, value):
    """Publish without replacement, then fsync file and directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b'\n'
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with open(temporary, 'xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError('Immutable ledger collision: ' + str(path))
    finally:
        temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def validate_row(row):
    if any(k not in row for k in FIELDS):
        raise ValueError('Incomplete score row')
    for key in FIELDS[1:]:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid score ' + key)
        if value < 0 or (key not in ('generated_tokens', 'search_calls') and value > 1):
            raise ValueError('Out of range score ' + key)
    for key in ('generated_tokens', 'search_calls'):
        if int(row[key]) != row[key]:
            raise ValueError('Non-integral count')

def identities(manifest):
    return {k:manifest[k] for k in ('config_sha256', 'dataset_sha256', 'model_sha256')}

def read_prefix(path, manifest, sources=None):
    value = json.loads(Path(path).read_text())
    if any(value.get(k) != v for k,v in identities(manifest).items()):
        raise ValueError('Preserved prefix identity mismatch')
    rows = value['rows']
    if len(rows) > manifest['dataset_count']:
        raise ValueError('Prefix larger than final set')
    for index,row in enumerate(rows):
        validate_row(row)
        if row.get('index', index) != index:
            raise ValueError('Prefix order mismatch')
        if sources is not None and row['source'] != sources[index]:
            raise ValueError('Prefix source ordering mismatch')
    return rows

def committed_batches(output, manifest, prefix_count):
    result = {}
    for done in sorted((Path(output)/'batches').glob('*.done.json')):
        receipt = json.loads(done.read_text())
        batch = done.with_name(done.name.replace('.done.json', '.json'))
        if digest(batch) != receipt['sha256']:
            raise ValueError('Batch SHA mismatch')
        value = json.loads(batch.read_text())
        if value['identity'] != identities(manifest) or value['prefix_count'] != prefix_count:
            raise ValueError('Batch identity mismatch')
        if not value['rows']:
            raise ValueError('Empty batch')
        indices = [r['index'] for r in value['rows']]
        if indices != list(range(indices[0], indices[0]+len(indices))):
            raise ValueError('Batch is not contiguous')
        for row in value['rows']:
            validate_row(row)
            index = row['index']
            if index < prefix_count or index >= manifest['dataset_count'] or index in result:
                raise ValueError('Duplicate/outside range question index')
            result[index] = row
    return result

def merge(output, prefix, manifest):
    rows = read_prefix(prefix, manifest)
    rest = committed_batches(output, manifest, len(rows))
    if set(rest) != set(range(len(rows), manifest['dataset_count'])):
        raise ValueError('Incomplete evaluation: missing question indices')
    rows = rows + [rest[i] for i in range(len(rows), manifest['dataset_count'])]
    # Bind every saved score to its actual dataset question, beyond aggregate counts.
    import pandas as pd
    from eval_worker import question_sha
    frame=pd.concat([pd.read_parquet(x['path']) for x in manifest['dataset_files']])
    if len(frame) != len(rows):
        raise ValueError('Dataset changed cardinality')
    for index,row in enumerate(rows):
        if row['source'] != frame.iloc[index]['data_source']:
            raise ValueError('Saved score source order mismatch')
        if index in rest and row.get('question_sha256') != question_sha(frame.iloc[index], manifest['prompt_key']):
            raise ValueError('Saved score question identity mismatch')
    if dict(Counter(r['source'] for r in rows)) != manifest['expected_source_counts']:
        raise ValueError('Final seven-source counts mismatch')
    from search_r1.baselines.evaluation import aggregate_answer_metrics
    metrics = aggregate_answer_metrics(rows)
    for source in manifest['expected_source_counts']:
        group = [r for r in rows if r['source'] == source]
        metrics['val/test_score/'+source] = sum(r['em'] for r in group)/len(group)
        metrics['val/evaluated_count/'+source] = len(group)
    execution_records=[]
    provenance_paths=sorted(Path(output).glob('rank-*-provenance.json'))
    if {p.name for p in provenance_paths} != {'rank-%d-provenance.json'%rank for rank in range(8)}:
        raise ValueError('Eight rank execution provenance records required')
    for provenance in provenance_paths:
        value=json.loads(provenance.read_text())
        if value['identity'] != identities(manifest):
            raise ValueError('Rank execution provenance identity mismatch')
        if value['execution'] not in execution_records:
            execution_records.append(value['execution'])
    if len(execution_records) != 1:
        raise ValueError('Ranks disagree on reviewed execution/resource protocol')
    changes=['8 resident TP1 ranks', 'local batch 64 (global scheduling width 512)',
             'no FSDP/reload/cache teardown', 'no eight-GPU rollout padding']
    if any(x.get('allocation_kind')=='shared_systemd' for x in execution_records):
        changes.extend(['explicit shared systemd memory_headroom_shared_v1 execution',
                        'physical GPUs 8; Slurm reserved GPUs 0; no exclusive allocation claim',
                        'resident vLLM GPU memory/cache fraction 20%'])
    result = dict(identity=identities(manifest), evaluated_count=len(rows), metrics=metrics,
        execution_changes=changes,execution_records=execution_records,
        numerical_caveat='HF weights unchanged; altered greedy batching/kernel shapes are not guaranteed bit-identical.')
    atomic_immutable(Path(output)/'final-metrics.json', result)
    atomic_immutable(Path(output)/'final-metrics.done.json', dict(sha256=digest(Path(output)/'final-metrics.json')))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare-manifest', action='store_true')
    p.add_argument('--config'); p.add_argument('--model'); p.add_argument('--prefix')
    p.add_argument('--output', required=True); p.add_argument('--manifest')
    args=p.parse_args()
    if args.prepare_manifest:
        atomic_immutable(args.output, asset_manifest(args.config,args.model))
    else:
        print(json.dumps(merge(args.output,args.prefix,json.loads(Path(args.manifest).read_text())),sort_keys=True))
