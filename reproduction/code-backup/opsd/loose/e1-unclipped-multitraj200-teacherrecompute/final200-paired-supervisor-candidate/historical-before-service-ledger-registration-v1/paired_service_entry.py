#!/usr/bin/env python3
"""Read-only loaded retriever observation before the original uvicorn.run.

No index/model construction or query is added. The original be0 neutral entry
and c5bf retriever_local/server perform all construction and serving unchanged.
"""
import argparse
import hashlib
import os
from pathlib import Path
import runpy
import sys
import types

ORIGINAL = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/tools/formal-71d2501d4512/retriever_entry.py'
ORIGINAL_SHA = 'be0a6228de6affa84b6cb0f2106ea8309d159991c0c37aefe31941bbdf548d60'


def parent(pin):
    # Reuse the adapter's independently bootstrapped exact-byte loader, not its
    # production main or native CPU checks after this service initializes CUDA.
    path=Path(__file__).resolve().parent/'paired_cpu_prepare.py'
    # Parent already same-FD compiled this entry; this neutral loader is pinned
    # through the same signed request's full control-tool closure.
    import stat
    before=path.lstat();fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd);raw=b''
        while True:
            part=os.read(fd,1024**2)
            if not part:break
            raw+=part
        ended=os.fstat(fd)
    finally:os.close(fd)
    after=path.lstat()
    fields=('st_dev','st_ino','st_uid','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
    if path.resolve(strict=True)!=path or not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid() or before.st_nlink!=1 or any(getattr(before,k)!=getattr(v,k) for v in (opened,ended,after) for k in fields):raise ValueError('Adapter physical bytes differ')
    module=types.ModuleType('paired_service_cpu_loader');module.__file__=str(path)
    # We execute only supervisor() after verifying the adapter pin supplied by
    # the launcher; pin here is a tuple, not an arbitrary code argument.
    if hashlib.sha256(raw).hexdigest()!=pin[0]:raise ValueError('Adapter fullSHA differs before execution')
    exec(compile(raw,str(path),'exec',dont_inherit=True,optimize=0),module.__dict__)
    return module.supervisor(pin[1])


def loaded_record(retriever,config,identity,faiss):
    if config.faiss_gpu is not True or config.encoder_device!='cuda' or config.retrieval_topk!=3:raise ValueError('Actual retriever GPU/loopback top3 config differs')
    index=retriever.index
    if not callable(getattr(index,'count',None)) or not callable(getattr(index,'at',None)):raise ValueError('Actual FAISS GPU shard view unavailable; do not infer GPU placement from flag')
    count=index.count()
    if type(count) is not int or count!=8:raise ValueError('Actual FAISS shard cardinality differs')
    shards=[]
    for i in range(count):
        shard=faiss.downcast_index(index.at(i))
        if not callable(getattr(shard,'getDevice',None)):raise ValueError('Actual GPU shard device view unavailable')
        device=shard.getDevice()
        if type(device) is not int:raise ValueError('Actual GPU shard device not typed int')
        shards.append(dict(ordinal=i,device=device,type_module=type(shard).__module__,type_name=type(shard).__qualname__,d=int(shard.d),ntotal=int(shard.ntotal)))
    devices=sorted(v['device'] for v in shards)
    if devices!=list(range(8)):raise ValueError('Actual GPU shard device set differs')
    model=retriever.encoder.model
    parameter_devices=sorted({str(p.device) for p in model.parameters()})
    if parameter_devices!=['cuda:0']:raise ValueError('E5 actual CUDA device differs')
    return dict(version=1,kind='ACTUAL_LOADED_SAME_PROCESS_GPU_FAISS_AND_E5_OBSERVATION',process=identity,
        gpu_index=dict(type_module=type(index).__module__,type_name=type(index).__qualname__,count=count,devices=devices,shards=shards),
        e5=dict(type_module=type(model).__module__,type_name=type(model).__qualname__,parameter_devices=parameter_devices,
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),encoder_device=retriever.encoder.device),
        config=dict(faiss_gpu=config.faiss_gpu,encoder_device=config.encoder_device,retrieval_topk=config.retrieval_topk,
            index_path=config.index_path,corpus_path=config.corpus_path,retrieval_model_path=config.retrieval_model_path,
            retrieval_use_fp16=config.retrieval_use_fp16),scope='ACTUAL_OBJECT_PLACEMENT_BEFORE_ORIGINAL_SERVER_NO_ADDED_FORWARD_OR_RETRIEVAL')


def main():
    p=argparse.ArgumentParser();p.add_argument('--supervisor-sha256',required=True);p.add_argument('--adapter-sha256',required=True);p.add_argument('--run',required=True)
    a,original_args=p.parse_known_args();original_args=original_args[1:] if original_args[:1]==['--'] else original_args
    s=parent((a.adapter_sha256,a.supervisor_sha256));run=Path(a.run);owner=s.allocation()
    s.same(run,s.ROOT/'paired-final200-runs'/('eval-'+owner['job_id']),'Service run scope differs')
    prepared,_=s.reference(dict(path=str(run/'cpu-prepared.json'),sha256=s.capture(run/'cpu-prepared.json')[1]['sha256']),root=run)
    for rec in prepared['source_members'].values():s.capture(rec['physical_path'],rec['sha256'])
    import uvicorn,faiss
    original_run=uvicorn.run;called=False
    def observe(app,**kwargs):
        nonlocal called
        if called:raise ValueError('Unexpected repeated service start')
        called=True
        # Locate only the exact already-qualified c5bf server frame. Do not
        # evaluate arbitrary frame content or call any extra model operation.
        frame=sys._getframe(1);target=s.REPO/'search_r1/search/retrieval_server.py'
        while frame is not None and Path(frame.f_code.co_filename)!=target:frame=frame.f_back
        if frame is None:raise ValueError('Original frozen server frame absent')
        record=loaded_record(frame.f_globals['retriever'],frame.f_globals['config'],s.proc_identity(os.getpid(),owner['job_id'],owner['step_id']),faiss)
        if kwargs.get('host')!='127.0.0.1' or kwargs.get('port')!=18084:raise ValueError('Actual server listener config differs')
        s.write_once(run/'actual-loaded-retriever.json',record)
        return original_run(app,**kwargs)
    uvicorn.run=observe
    raw,_=s.capture(ORIGINAL,ORIGINAL_SHA,keep=True)
    sys.argv=[ORIGINAL]+original_args
    try:exec(compile(raw,ORIGINAL,'exec',dont_inherit=True,optimize=0),{'__name__':'__main__','__file__':ORIGINAL})
    finally:uvicorn.run=original_run


if __name__=='__main__':main()
