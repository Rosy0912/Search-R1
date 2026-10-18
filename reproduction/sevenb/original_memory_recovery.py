"""One inspected host-memory replacement; the scientific runner is unchanged."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from .contract import ROOT, validate_plan, verify_data, verify_model
from reproduction.seven_dataset.contracts import sha

JOB = 'zhangjunming-7b-original-mem-v1'
RUN = ROOT/'runs/original-hostmem-v1'
ALLOCATION = dict(gpus=8, cpus=64, memory_gib=900, time='7-00:00:00')
OLD_SOURCE = 'c65bfee308d758707325992e3c6267f43a65f9ef'


def validate(plan, old):
    from omegaconf import OmegaConf
    validate_plan(old)
    assert old['kind']=='original' and old['repeat_index']==0
    assert plan['protocol']=='sevenb_original_host_memory_replacement_v1'
    assert plan['job']==JOB and plan['root']==str(RUN)
    assert plan['output']==str(RUN/'output') and plan['allocation']==ALLOCATION
    assert plan['old_job']=='139101' and plan['old_source']==OLD_SOURCE
    assert plan['fresh_from_common_initialization'] is True
    assert plan['resume_from_partial_state'] is False
    a=OmegaConf.to_container(OmegaConf.create(old['config_text']),resolve=True)
    b=OmegaConf.to_container(OmegaConf.create(plan['config_text']),resolve=True)
    assert b['trainer']['default_local_dir']==str(RUN/'checkpoints')
    b['trainer']['default_local_dir']=a['trainer']['default_local_dir']
    assert a==b, 'Only the checkpoint output path may change'
    for k in ('model','model_label','model_revision','model_files','checkpoint_identity',
              'data_root','data_manifest_sha256','ordered_ids_sha256','final_content_identity',
              'retriever_receipt','retriever_receipt_sha256','retriever','profile','kind',
              'repeat_index','repeat_count','outer'):
        assert plan[k]==old[k],k
    return RUN


def verify(plan):
    oldfile=Path(plan['old_plan']);assert sha(oldfile)==plan['old_plan_sha256']
    old=json.loads(oldfile.read_text());root=validate(plan,old)
    assert sha(plan['failure_proof'])==plan['failure_proof_sha256']
    failure=json.loads(Path(plan['failure_proof']).read_text())
    assert failure['job']=='139101' and failure['complete_native_manifest_absent']
    assert failure['status']=='OUT_OF_MEMORY_DURING_FULL_STATE_SAVE_AT5_NO_COMPLETE_RESUME'
    assert not subprocess.check_output(['squeue','-h','-j','139101'],text=True).strip()
    rows=subprocess.check_output(['sacct','-X','-n','-P','-j','139101',
        '--format=JobIDRaw,JobName%100,State,ElapsedRaw,ExitCode,AllocTRES%250'],text=True).strip().split('|')
    assert rows[:3]==['139101','zhangjunming-7b-original-0','OUT_OF_MEMORY']
    assert int(rows[3])==failure['elapsed_seconds'] and rows[4]=='0:125' and 'gres/gpu=8' in rows[5]
    for n,h in plan['wrapper_files'].items():assert sha(Path(plan['source'])/n)==h,n
    # All algorithm, optimizer, snapshot and model code stays byte-identical.
    prior=json.loads(Path(old['source_manifest']).read_text())
    for n,h in prior.items():
        if n.startswith(('search_r1/','verl/','release/')):
            assert sha(Path(plan['source'])/n)==h,n
    assert verify_model('base')==plan['model_files'];verify_data()
    assert sha(plan['retriever_receipt'])==plan['retriever_receipt_sha256']
    return root


def run(planfile):
    plan=json.loads(Path(planfile).read_text());root=verify(plan)
    sub=json.loads((root/'submission.json').read_text());jid=os.environ['SLURM_JOB_ID']
    assert sub['job_id']==jid and sub['job']==JOB and sub['plan_sha256']==sha(planfile)
    row=subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True)
    for expected in ('JobName='+JOB+' ','UserId=jmzhang(1066)','gres/gpu=8','mem=900G','cpu=64'):
        assert expected in row,expected
    assert os.environ['SLURM_JOB_NODELIST'] in ('node8','node9')
    inv=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(inv)==8 and len({r.split(',')[0] for r in inv})==8
    assert all(float(r.split(',')[-1])>=44000 for r in inv)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True).strip()
    with (root/'entry-once.json').open('x') as f:
        json.dump(dict(at=time.time(),job_id=jid,gpus=inv,plan_sha256=sha(planfile),allocation=ALLOCATION),f)
    from omegaconf import OmegaConf
    import socket
    cfg=OmegaConf.create(plan['config_text']);cfg.baselines.local_runtime_host=socket.gethostname().split('.')[0]
    from search_r1.baselines.controller import validate_config
    validate_config(cfg)
    from .entry import run_original
    run_original(plan,root,cfg)


if __name__=='__main__':run(sys.argv[1])
