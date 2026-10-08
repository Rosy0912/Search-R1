import subprocess,json,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
def run(args):
 p=subprocess.run(args,text=True,capture_output=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
before=run(['scontrol','show','job','3222'])
assert before['returncode']==0
assert 'JobName=bapo-instruct300-formal300' in before['stdout'] and f'WorkDir={root}' in before['stdout'] and 'NodeList=h100-5' in before['stdout']
p=root/'logs'/'user-stop3222-evidence.json'
record={'reason':'User explicitly requested repair then stop current training','before':before}
p.write_text(json.dumps(record,ensure_ascii=False,indent=2))
record['cancel']=run(['scancel','3222']);assert record['cancel']['returncode']==0
for _ in range(20):
 record['after']=run(['scontrol','show','job','3222'])
 if 'JobState=CANCELLED' in record['after']['stdout'] or 'JobState=COMPLETED' in record['after']['stdout']:break
 time.sleep(1)
record['gpu_processes']=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
p.write_text(json.dumps(record,ensure_ascii=False,indent=2));print(json.dumps(record,ensure_ascii=False,indent=2))
