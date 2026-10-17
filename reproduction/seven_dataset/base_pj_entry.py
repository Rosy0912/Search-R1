import argparse,json
from pathlib import Path
from .base_pj import inspect_pj_base
from .entry import run_frozen

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());cfg,rows,root=inspect_pj_base(plan,require_retired=not a.preflight_only)
    if a.preflight_only:
        print(json.dumps(dict(status='PJ_BASE_PREFLIGHT_ONLY_NO_ALLOCATION',questions=len(rows))));return
    run_frozen(plan,a.plan,cfg,rows,root)

if __name__=='__main__':main()
