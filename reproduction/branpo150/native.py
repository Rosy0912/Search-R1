"""Bound the real140/150 native checkpoint; no scheduler or GPU mutation."""
import ast
import base64
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from .contract import sha, PARENTS, ROOT, canonical
from .state import audit_native


def small_file(path):
    p=Path(path);raw=p.read_bytes();assert len(raw)<8*1024**2
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                base64=base64.b64encode(raw).decode())


def metadata(parent,size,label=240):
    root=Path(parent['root']);native=root/'checkpoints'/f'global_step_{label}'
    sub=json.loads((root/'submission.json').read_text())
    assert sub['job_id']==PARENTS[size][0] and sub['plan_sha256']==PARENTS[size][2]
    rows=subprocess.check_output(['sacct','-n','-P','-X','-j',sub['job_id'],
                                  '--format=JobID,JobName%90,State,ExitCode,ElapsedRaw,AllocTRES%180'],text=True).strip().splitlines()
    rows=[r.split('|') for r in rows if r.split('|')[0]==sub['job_id']]
    assert len(rows)==1 and rows[0][1]==sub['job']
    progress=json.loads((root/'audit/progress.json').read_text())
    paths=[native/'actor'/f'{k}_world_size_8_rank_{r}.pt' for r in range(8)
           for k in ('model','optim','extra_state')]+[native/'data.pt',native/'actor/fsdp_config.json']
    return dict(at=time.time(),scheduler=rows[0],progress=progress,native=str(native),
                complete=all(p.is_file() and p.stat().st_size>0 for p in paths),
                metadata={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                          for p in paths if p.is_file()})


def next_batch(parent,data_state,label):
    import torch
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader
    import verl.trainer.main_ppo as module
    source=inspect.getsource(module.create_rl_sampler)
    assert hashlib.sha256(source.encode()).hexdigest()=='ac61f3bda6d012d2420754e5534314ada77f54bdf90cedda9b7157a5507a1981'
    cfg=OmegaConf.load(Path(parent['root'])/'resolved.yaml')
    assert cfg.data.shuffle is True and cfg.data.seed is None
    assert cfg.data.sampler is None or cfg.data.sampler.get('class_path') is None
    assert cfg.data.train_batch_size==256 and cfg.data.dataloader_num_workers==8
    data=range(169028);sampler=module.create_rl_sampler(cfg.data,data)
    assert sampler.generator.initial_seed()==67280421310721
    loader=StatefulDataLoader(data,batch_size=256,num_workers=8,drop_last=True,sampler=sampler)
    main=data_state['_snapshot']['_main_snapshot']
    assert data_state['_snapshot']['_snapshot_step']==label
    assert main['_sampler_iter_state']=={'samples_yielded':label*256}
    assert main['_sampler_iter_yielded']==label and main['_num_workers']==8
    assert data_state['_steps_since_snapshot']==0 and data_state['_iterator_finished'] is False
    loader.load_state_dict(data_state);it=iter(loader);indices=next(it).tolist()
    receipt=json.loads(Path(parent['converted_data_receipt']).read_text())['splits']['train']
    assert sha(receipt['tasks'])==receipt['tasks_sha256']
    assert sha(receipt['verl'])==receipt['verl_sha256']
    table=pq.read_table(receipt['tasks'],columns=['index'])
    assert table.num_rows==169028
    ids=[str(table['index'][i].as_py()) for i in indices]
    assert len(ids)==len(set(ids))==256
    # Explicitly terminate these disposable CPU index-loader workers.
    if hasattr(it,'_shutdown_workers'):it._shutdown_workers()
    return dict(state=data_state,next_label=label+1,next_indices=indices,next_ids=ids,
                sampler_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                sampler_generator_initial_seed=67280421310721,
                full_train_tasks_sha256=receipt['tasks_sha256'],full_train_verl_sha256=receipt['verl_sha256'])


def run_parent140(parent_plan,size,output):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from reproduction.official_branpo.entry import verify as verify_parent
    assert not torch.cuda.is_initialized()
    assert sha(parent_plan)==PARENTS[size][2]
    parent=json.loads(Path(parent_plan).read_text());verify_parent(parent)
    actual=metadata(parent,size)
    assert actual['scheduler'][2] in ('RUNNING','COMPLETED')
    assert actual['progress']['completed_actual_iterations']>=140 and actual['complete']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'execution-once.json').open('x').write(canonical(dict(at=time.time(),size=size,parent_job_id=PARENTS[size][0],no_auto_retry=True)))
    native=Path(actual['native']);fsdp=json.loads((native/'actor/fsdp_config.json').read_text())
    assert fsdp=={'FSDP_version':1,'world_size':8}
    hf=native/'actor/huggingface';config=AutoConfig.from_pretrained(hf,local_files_only=True)
    assert config.model_type=='qwen2' and config.vocab_size==(151936 if size=='3b' else 152064)
    assert (config.hidden_size,config.num_hidden_layers)==((2048,36) if size=='3b' else (3584,28))
    with torch.device('meta'):
        model=AutoModelForCausalLM.from_config(config)
    shapes={k:list(v.shape) for k,v in model.state_dict().items()}
    physical_numel=sum(p.numel() for p in model.parameters());del model
    expected=audit_native(native/'actor',shapes,physical_numel,140,size)
    small={}
    for p in [Path(parent_plan),Path(parent['root'])/'submission.json',Path(parent['root'])/'resolved.yaml',
              native/'data.pt',native/'actor/fsdp_config.json',*sorted(hf.iterdir())]:
        if p.is_file():small[str(p)]=small_file(p)
    data_state=torch.load(native/'data.pt',map_location='cpu',weights_only=False)
    dloader=next_batch(parent,data_state,240)
    again=metadata(parent,size)
    assert actual['metadata']==again['metadata'], 'Native changed while inspected'
    proof=dict(at=time.time(),size=size,parent_job_id=PARENTS[size][0],parent_plan_sha256=PARENTS[size][2],
               native=expected,dataloader=dloader,small_files=small,parent_scheduler=actual['scheduler'],
               target_actual_updates=150,remaining_updates=10,
               original_engine_does_not_persist_uninterrupted_rollout_rng=True,
               no_bitwise_uninterrupted_trajectory_claim=True,no_gpu_action=True)
    file=output/'native140-proof.json'
    with file.open('x') as stream:json.dump(proof,stream,indent=2,allow_nan=False)
    print('NATIVE140_PROOF='+json.dumps(dict(path=str(file),sha256=sha(file),proof=proof)),flush=True)
    assert not torch.cuda.is_initialized()
