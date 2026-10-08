#!/usr/bin/env python3
"""New-source NCCL scalar routing gate. Inputs remain CPU on every rank."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import torch
import torch.distributed as dist
repo=Path(os.environ['SEARCH_R1_REPO']).resolve()
expected=os.environ['E1_EXPECTED_SOURCE']
source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
assert source==expected and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
sys.path.insert(0,str(repo))
from search_r1.baselines.distillation import global_token_count, global_token_denominator
from search_r1.baselines.objectives import row_denominator
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local);dist.init_process_group('nccl')
assert dist.get_world_size()==8 and dist.get_backend()=='nccl'
try:
    cases=[('one_empty_rank',[0,1,2,3,4,5,6,7],[3,4,5,6,7,8,9,10]),
           ('global_empty_tokens',[0]*8,[3,4,5,6,7,8,9,10]),
           ('unequal_rows_with_zero',[2,1,4,1,6,2,8,3],[0,2,3,4,5,6,7,8]),
           ('global_empty_rows',[0]*8,[0]*8)]
    records=[]
    for name,tokens,rows in cases:
        mask=torch.ones(tokens[rank],device='cpu');before=mask.clone()
        total,world=global_token_count(mask)
        den=global_token_denominator(mask)
        row=row_denominator(rows[rank],torch.device('cpu'))
        assert total.device==den.device==row.device==torch.device('cuda',local)
        assert world==8 and total.item()==sum(tokens)
        assert den.item()==max(sum(tokens),1)/8 and row.item()==max(sum(rows),1)/8
        assert mask.device.type=='cpu' and torch.equal(mask,before)
        records.append(dict(case=name,local_tokens=tokens[rank],local_rows=rows[rank],
            global_tokens=total.item(),global_row_denominator=row.item(),
            token_denominator=den.item(),input_device='cpu',collective_device=str(total.device),world=world))
    output=Path(os.environ['E1_COUNT_FIXTURE_OUTPUT']).resolve();output.mkdir(parents=True,exist_ok=True)
    out=dict(status='PASS_NATIVE_NCCL_CPU_INPUT_COUNTS_ONLY',source_commit=source,rank=rank,
        world=8,records=records,torch_version=torch.__version__,
        source_files_sha256={f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in
            ('search_r1/baselines/distillation.py','search_r1/baselines/objectives.py')},
        fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/f'rank-{rank}.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print('NATIVE_NCCL_CPU_INPUT_COUNTS_JSON='+json.dumps(out,sort_keys=True,allow_nan=False),flush=True)
    dist.barrier()
finally:dist.destroy_process_group()
