"""Dedicated allocated-process launcher; acquire existing GPU idle/lease checks."""
import argparse
import os
import subprocess
import sys
from .contracts import require


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        require(os.environ.get('SEARCH_R1_EXECUTION_BACKEND')=='pj_pt','Only PT provides a restricted container GPU namespace')
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines()
        require(len(rows)==8,'Expected exactly eight scheduler-allocated devices')
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(x.strip() for x in rows)
    args=[sys.executable,'-m','search_r1.baselines.preflight','--',sys.executable,
          '-m','reproduction.seven_dataset.entry','--plan',a.plan]
    os.execv(sys.executable,args)

if __name__=='__main__':main()
