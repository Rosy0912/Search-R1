import argparse,os,subprocess,sys
from .contracts import require

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    require(os.environ.get('SEARCH_R1_EXECUTION_BACKEND')=='pj_pt' and not os.environ.get('SLURM_JOB_ID'),'Wrong backend')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines()
    require(len(rows)==8,'Expected eight allocated devices');os.environ['CUDA_VISIBLE_DEVICES']=','.join(r.strip() for r in rows)
    os.execv(sys.executable,[sys.executable,'-m','search_r1.baselines.preflight','--',sys.executable,'-m','reproduction.seven_dataset.base_pj_entry','--plan',a.plan])

if __name__=='__main__':main()
