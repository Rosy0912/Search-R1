import argparse,json
from pathlib import Path
from .ablation_pj import inspect
from .entry import run_frozen

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());cfg,rows,root=inspect(plan)
    if a.preflight_only:print(json.dumps(dict(status='PJ_ABLATION_INPUTS_VERIFIED',count=len(rows))));return
    run_frozen(plan,a.plan,cfg,rows,root)

if __name__=='__main__':main()
