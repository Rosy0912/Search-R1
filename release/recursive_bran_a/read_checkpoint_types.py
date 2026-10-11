"""One CPU-only checkpoint type/step inspection; no restore or model forward."""
import argparse, hashlib, json, os, time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()

def signature(p):
    s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--expected-step',type=int,required=True)
    p.add_argument('--expected-adam',type=int,required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert sha(a.manifest)==a.manifest_sha256
    r=json.loads(a.manifest.read_text());i=r['identity']
    assert i['config_sha256']==a.config_sha256 and i['code_commit']==a.source_commit and i['completed_step']==a.expected_step
    assert r['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and len(r['worker_states'])==8
    import torch
    torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    reports=[]
    for w in sorted(r['worker_states'],key=lambda w:w['identity']['rank']):
        path=Path(w['path']);assert path.parent.resolve()==a.manifest.parent.resolve()
        before=signature(path);assert before[2]==w['bytes'] and sha(path)==w['sha256']
        v=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert v['identity']==w['identity']
        dtypes={str(t.dtype) for t in v['parameters'].values()};assert dtypes=={'torch.float32'}
        states=v['optimizer']['state'];steps=set();moments=set()
        for state in states.values():
            steps.add(int(state['step'].item()) if isinstance(state['step'],torch.Tensor) else int(state['step']))
            moments.update(str(state[k].dtype) for k in ('exp_avg','exp_avg_sq'))
        assert steps=={a.expected_adam} and moments=={'torch.float32'}
        assert len(states)==w['optimizer_state_entries'] and v['scheduler']['last_epoch']==a.expected_step
        assert signature(path)==before and not torch.cuda.is_initialized()
        reports.append(dict(rank=w['identity']['rank'],parameters=len(v['parameters']),parameter_numel=sum(t.numel() for t in v['parameters'].values()),
            parameter_dtype=sorted(dtypes),adam_entries=len(states),adam_steps=sorted(steps),moment_dtype=sorted(moments),
            scheduler_epoch=v['scheduler']['last_epoch'],file_sha256=w['sha256'],fresh_file_sha_accepted=True))
        del v
    assert {x['rank'] for x in reports}==set(range(8))
    print(json.dumps(dict(timestamp=time.time(),accepted=True,manifest_sha256=a.manifest_sha256,workers=reports,
        cuda_initialized=torch.cuda.is_initialized(),scope='Independent full-file SHA plus CPU mmap dtype/Adam-step/scheduler inspection; no full tensor arithmetic or forward/restore'),sort_keys=True))
