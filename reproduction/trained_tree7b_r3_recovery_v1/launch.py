"""Acquire process leases before starting the trained model's inference entry."""
import argparse
import os
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    assert os.environ.get('SEARCH_R1_EXECUTION_BACKEND')=='pj_pt' and not os.environ.get('SLURM_JOB_ID')
    indices=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines()
    assert len(indices)==8
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(r.strip() for r in indices)
    os.execv(sys.executable,[sys.executable,'-m','search_r1.baselines.preflight','--',sys.executable,
                            '-m','reproduction.trained_tree7b_r3_recovery_v1.entry','--plan',a.plan])


if __name__=='__main__':main()
