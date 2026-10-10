"""Repair four-GPU cardinality in the source-bound safety policy, only while idle."""
import ast,json,pathlib,re,subprocess
from finalize_topd200_draft import BASE,ROOT,sha,main as refresh

def main():
 if 'topd' in subprocess.check_output(['squeue','-u','mou176','-h','-o','%j'],universal_newlines=True).lower():raise RuntimeError('Active TOP-D job')
 policy=ROOT/'tools/qwen25-32b-window-safety-direct-own-sm-v2'
 for n in ['startup_guard.py','runtime_safety.py']:
  p=policy/n;s=p.read_text();s=re.sub(r'== 8(?![0-9])','== 4',s);ast.parse(s);p.write_text(s)
 p=policy/'files.json';j=json.loads(p.read_text());j['files']={str(f.relative_to(policy)):{'size':f.stat().st_size,'sha256':sha(f)} for f in policy.rglob('*') if f.is_file() and f!=p};p.write_text(json.dumps(j,indent=2,sort_keys=True))
 cfg=BASE/'outputs/training-preparation/hpc3-topd200-protocol-draft.json';c=json.loads(cfg.read_text());old=c['telemetry_policy'];new=dict(old,runtime_sha256=sha(policy/'runtime_safety.py'),guard_sha256=sha(policy/'startup_guard.py'),manifest_sha256=sha(p))
 guardian=ROOT/'tools/topd-instruct300-launch-v6/runtime/window_guardian.py';s=guardian.read_text()
 for key in ['runtime_sha256','guard_sha256','manifest_sha256']:s=s.replace(old[key],new[key])
 ast.parse(s);guardian.write_text(s);c['telemetry_policy']=new;cfg.write_text(json.dumps(c,indent=2))
 refresh();print('FOUR_GPU_SOURCE_BOUND_SAFETY_CARDINALITY_FIXED')
if __name__=='__main__':main()
