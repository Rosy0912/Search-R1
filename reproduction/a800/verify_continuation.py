"""Read-only, segment-local gate: five real new updates and source-to-new delta."""
import hashlib
import json
import math
from pathlib import Path
import re
import torch
from safetensors import safe_open
from verl.utils.checkpoint_integrity import validate_saved_checkpoint

BASE = Path('/data/search-r1')


def verify(manifest):
    k=manifest['completed_updates']
    assert type(k) is int and 0<k<=495
    assert manifest['optimizer_reset'] is True
    source=Path(manifest['actor_checkpoint'])
    canonical=Path(manifest['reference_model'])
    assert source.resolve()!=canonical.resolve()
    assert source.name=='global_step_'+str(k)
    assert manifest['acceptance_step']==k+5
    started=manifest['run_started']
    log=Path(manifest['train_log'])
    assert log.stat().st_mtime>=started
    steps={}
    text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',log.read_text(errors='replace'))
    run_id=manifest['run_id']
    assert isinstance(run_id,str) and run_id, 'Missing independent segment identity'
    assert text.splitlines()[0]=='SEARCH_R1_SEGMENT '+run_id, 'Wrong or stale segment log'
    assert manifest['source_archive_sha256']=='f5b9e0d987a7fcb352b1614e9ce4c6b5df1c9c78f3fc83a4fbe247b9f81f7f6a'
    assert manifest['provenance_verified'] is True
    # Bind the actual source and original reference files, not just their directory names.
    for root,key in ((source,'actor_file_hashes'),(canonical,'reference_file_hashes')):
        hashes=manifest[key]
        weights=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        required=set(weights.values()) | {'config.json','model.safetensors.index.json'}
        assert required<=set(hashes), 'Missing checkpoint provenance'
        for name,expected in hashes.items():
            file=(root/name).resolve()
            assert file.is_relative_to(root.resolve()), 'Provenance path escapes checkpoint'
            digest=hashlib.sha256()
            with file.open('rb') as reader:
                for chunk in iter(lambda:reader.read(8*1024*1024),b''): digest.update(chunk)
            assert digest.hexdigest()==expected, 'Changed checkpoint/reference file: '+name
    for line in text.splitlines():
        match=re.search(r'\bstep:(\d+) - (.*)',line)
        if not match: continue
        metrics={}
        for piece in match.group(2).split(' - '):
            if ':' not in piece: continue
            key,value=piece.rsplit(':',1)
            try: metrics[key]=float(value)
            except ValueError: pass
        if 'critic/rewards/mean' in metrics:
            step=int(match.group(1))
            assert step>k, 'Prior segment training metrics in new log'
            assert step not in steps, 'Duplicate actual update in segment'
            steps[step]=metrics
    window=range(k+1,k+6)
    assert all(i in steps for i in window), 'Five actual new updates not available'
    for i in window:
        assert all(math.isfinite(v) for v in steps[i].values())
        assert 0<=steps[i]['critic/rewards/mean']<=1
    # Revalidate source integrity; old training deltas cannot satisfy this gate.
    validate_saved_checkpoint(source,canonical,0)
    checkpoint=Path(manifest['checkpoint_dir'])/'actor'/('global_step_'+str(k+5))
    count=validate_saved_checkpoint(checkpoint,canonical,started)
    name='model.layers.0.self_attn.q_proj.weight'
    def tensor(directory):
        mapping=json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
        with safe_open(str(directory/mapping[name]),framework='pt',device='cpu') as reader:
            return reader.get_tensor(name).float()
    old,new=tensor(source),tensor(checkpoint)
    assert old.shape==new.shape and torch.isfinite(new).all()
    delta=(old-new).abs()
    assert (delta>0).any(), 'No new parameter change from transferred checkpoint'
    return dict(run_id=run_id,accepted_updates=5,completed_updates_before_segment=k,
                latest_logged_update=max(steps),checkpoint=str(checkpoint),
                validated_tensor_count=count,finite_all_saved_tensors=True,
                changed_elements=int((delta>0).sum()),max_abs_delta=float(delta.max()),
                metrics={str(i):steps[i] for i in window},
                metric_precision='console rounds to three decimals',
                checkpoint_semantics='weight continuation; fresh Adam, RNG/sampler not restored')


if __name__=='__main__':
    manifest=json.loads((BASE/'logs/a800-continuation-manifest.json').read_text())
    result=verify(manifest)
    (BASE/'logs/a800-five-resumed-updates-acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
