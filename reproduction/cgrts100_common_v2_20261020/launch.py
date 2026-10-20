"""Use the unchanged native eight-device process lease before the final entry."""
import os,subprocess,sys
from .contract import require

def main():
 args=sys.argv[1:];require(len(args)==2 and args[0]=='--plan','One bound plan required')
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines()
 require(len(rows)==8,'Eight allocated GPUs required')
 os.environ['CUDA_VISIBLE_DEVICES']=','.join(s.strip() for s in rows)
 os.execv(sys.executable,[sys.executable,'-m','search_r1.baselines.preflight','--',sys.executable,'-m','reproduction.cgrts100_common_v2_20261020.entry',*args])

if __name__=='__main__':main()
