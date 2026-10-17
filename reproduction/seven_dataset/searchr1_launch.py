"""Search-R1-only launcher; scheduler GPU lease checks precede frozen inference."""
import argparse
import os
import sys
from .contracts import require


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    require(os.environ.get('SEARCH_R1_EXECUTION_BACKEND')=='slurm_lumia'
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES')),'Baseline requires allocated Slurm GPUs')
    os.execv(sys.executable,[sys.executable,'-m','search_r1.baselines.preflight','--',
        sys.executable,'-m','reproduction.seven_dataset.searchr1_entry','--plan',a.plan])


if __name__=='__main__':main()
