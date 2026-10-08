"""Observe archived loader code on NEW common169028; never historical order proof."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace

GOAL_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_loader(reference, config, tokenizer):
    """Execute unchanged archive method with observable real dataset subclass."""
    import torch
    from omegaconf import OmegaConf, open_dict
    from verl.utils import model, fs
    provenance = json.loads((reference/'provenance.json').read_text())
    records = provenance['files']
    for name, entry in records.items():
        if 'snapshot' in entry:
            expected = entry.get('snapshot_sha256', entry.get('sha256'))
            if sha(reference/entry['snapshot']) != expected:
                raise ValueError('Modified archived reference snapshot: '+name)
    for name, module in [('verl/utils/model.py', model), ('verl/utils/fs.py', fs)]:
        if sha(module.__file__) != records[name]['runtime_sha256_required']:
            raise ValueError('Runtime baseline dependency differs: '+name)
    dataset = load_module('_archived_order_dataset', reference/'rl_dataset.py')
    dataset.verl_F = load_module('_archived_order_torch_functional', reference/'torch_functional.py')
    original = dataset.RLHFDataset
    class ObservedDataset(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observed_indices = []
        def __getitem__(self, index):
            row = super().__getitem__(index)
            self.observed_indices.append(int(index))
            return row
    dataset.RLHFDataset = ObservedDataset
    key = 'verl.utils.dataset.rl_dataset'
    previous = sys.modules.get(key)
    sys.modules[key] = dataset
    try:
        source = textwrap.dedent((reference/'create_dataloader.py').read_text())
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != '_create_dataloader':
            raise ValueError('Unexpected archived method snapshot')
        namespace = {'OmegaConf': OmegaConf, 'open_dict': open_dict}
        exec(compile(tree, str(reference/'create_dataloader.py'), 'exec'), namespace)
        holder = SimpleNamespace(config=config, tokenizer=tokenizer)
        namespace['_create_dataloader'](holder)
    finally:
        if previous is None:sys.modules.pop(key, None)
        else:sys.modules[key] = previous
    if holder.train_dataloader.num_workers != 0 or not holder.train_dataloader.drop_last:
        raise ValueError('Require actual single-process drop_last archive loader')
    return holder, provenance


def observe_epoch(loader, question_ids, rng_state, batch_size=512):
    import torch
    question_ids = list(question_ids)
    if len(set(question_ids)) != len(question_ids) or any(not isinstance(q,str) for q in question_ids):
        raise ValueError('Original stable question IDs must be unique strings')
    dataset = loader.dataset
    dataset.observed_indices.clear()
    # RNG state is captured/restored at the actual iterator boundary, not inferred
    # as the state of a model-initialized historical trainer.
    torch.set_rng_state(rng_state)
    initial = torch.get_rng_state().clone()
    orders, batches = [], []
    iterator = iter(loader)
    for ordinal, batch in enumerate(iterator):
        emitted = dataset.observed_indices[len(orders):]
        actual_ids = batch['index'].tolist()
        if len(emitted) != batch_size or actual_ids != [question_ids[i] for i in emitted]:
            raise ValueError('Actual collated IDs differ from observed dataset indices')
        orders.extend(emitted)
        batches.append(dict(batch_ordinal=ordinal, indices=emitted, question_ids=actual_ids))
    count = len(question_ids)//batch_size*batch_size
    if len(orders) != count or len(set(orders)) != count:
        raise ValueError('Capture must contain a complete unique drop_last epoch')
    manifest = dict(schema='baseline-aligned-order-v1', source='baseline-observed-indices',
        batch_size=batch_size, drop_last=True, num_workers=0, question_ids=question_ids,
        epochs=[orders], torch_version=str(torch.__version__),
        initial_rng_sha256=hashlib.sha256(bytes(initial.tolist())).hexdigest(),
        preiterator_cpu_rng_state=initial.tolist(), actual_batches=batches)
    return manifest


def write_exclusive(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary=tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    try:
        with os.fdopen(fd,'w') as handle:
            json.dump(value,handle,ensure_ascii=False,sort_keys=True,allow_nan=False)
            handle.write('\n');handle.flush();os.fsync(handle.fileno())
        os.link(temporary,path)
        os.chmod(path,0o444)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        os.unlink(temporary)


def capture(args):
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    sys.path.insert(0,str(Path(args.aligned_root).resolve()))
    from search_r1.baselines.holdout import _validate_prebuilt
    from search_r1.baselines.baseline_alignment_data import dataset_identity, verify_order_manifest
    common=Path(args.common_manifest).resolve()
    if sha(common)!=GOAL_SHA:raise ValueError('User-pinned original common manifest differs')
    config=OmegaConf.create(dict(data=dict(prompt_key='prompt',max_prompt_length=4096,
        train_batch_size=512,val_batch_size=512,shuffle_train_dataloader=True,
        train_data_num=None,val_data_num=None,return_raw_chat=False),
        trainer=dict(total_epochs=1,total_training_steps=201),
        actor_rollout_ref=dict(actor=dict(optim={})),critic=dict(optim={}),
        baselines=dict(holdout=dict(enabled=True,manifest_path=str(common),manifest_sha256=GOAL_SHA,
            data_root=str(common.parent),seed=42),bapo=dict(stage_validation_size=64))))
    bound=_validate_prebuilt(config)
    if bound['train_rows']!=169028:raise ValueError('New common reference must contain169028 training rows')
    config.data.train_files=[bound['effective_paths']['train']]
    config.data.val_files=[bound['effective_paths']['validation']]
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    reference=Path(args.reference).resolve()
    holder, provenance=reference_loader(reference,config,tokenizer)
    identity=dataset_identity(config.data.train_files)
    actual_ids=[row['index'] for row in holder.train_dataset.dataframe['extra_info']]
    if len(holder.train_dataset)!=169028 or actual_ids!=identity['question_ids']:
        raise ValueError('Real archived dataset filtering/order differs from common prepared training rows')
    if args.preiterator_rng:
        rng_bytes=json.loads(Path(args.preiterator_rng).read_text())
        if not isinstance(rng_bytes,list) or any(type(n)is not int or not 0<=n<=255 for n in rng_bytes):
            raise ValueError('Preiterator RNG must be explicit uint8 integer bytes')
        rng=torch.tensor(rng_bytes,dtype=torch.uint8)
        policy='supplied-preiterator-cpu-rng-with-actual-loader-observation'
    else:
        torch.manual_seed(42);rng=torch.get_rng_state().clone()
        policy='NEW-common169028-CPU-reference-seed42-at-preiterator-not-historical-model-RNG'
    manifest=observe_epoch(holder.train_dataloader,actual_ids,rng)
    if len(manifest['actual_batches'])!=330:raise ValueError('Expected330 complete512-question batches')
    manifest.update(reference_scope='new-common169028-reference-not-historical169103-equality',
        reference_source=provenance,reference_provenance_sha256=sha(reference/'provenance.json'),
        rng_policy=policy,supplied_rng_file_sha256=sha(args.preiterator_rng) if args.preiterator_rng else None,common_manifest_path=str(common),common_manifest_sha256=GOAL_SHA,
        train_semantic_sha256=identity['semantic_sha256'],train_files=identity['files'],
        tokenizer_root=str(Path(args.model).resolve()),tokenizer_files={str(p.relative_to(args.model)):sha(p)
            for p in sorted(Path(args.model).glob('*')) if p.is_file() and p.suffix in ('.json','.model','.txt')},
        capture_tool_sha256=sha(__file__),effective_loader_config=OmegaConf.to_container(config,resolve=True))
    manifest['order_sha256']=object_hash(manifest)
    verify_order_manifest(manifest,actual_ids,512)
    write_exclusive(args.output,manifest)
    return dict(path=str(Path(args.output).resolve()),sha256=sha(args.output),order_sha256=manifest['order_sha256'],
        rows=169028,batches=330,emitted_questions=168960,dropped_questions=68,rng_policy=policy)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aligned-root',required=True)
    parser.add_argument('--common-manifest',required=True)
    parser.add_argument('--model',required=True,help='Local official tokenizer/model directory; offline only')
    parser.add_argument('--reference',default=str(Path(__file__).parent/'reference'))
    parser.add_argument('--preiterator-rng',help='Optional JSON uint8 CPU RNG bytes captured before a reference iterator')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(capture(args),sort_keys=True))

if __name__=='__main__':main()
