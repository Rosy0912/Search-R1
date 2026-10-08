P='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304/evidence/first5-actual-review-v1'
import pathlib,json,os,datetime
p=pathlib.Path(P);d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
for name in ('producer-start.original.json','producer-exit.original.json','unsigned-step5-payload.json','receipt-preparation-audit.json','producer.stdout.original','producer.stderr.original'):
 f=p/name
 if f.exists():
  d['files'][name]={'bytes':f.stat().st_size}
  if name.endswith('original.json'):d['files'][name]['value']=json.loads(f.read_bytes())
if (p/'producer-start.original.json').exists():
 pid=json.loads((p/'producer-start.original.json').read_bytes())['child_pid'];q=pathlib.Path('/proc')/str(pid);d['child_proc_alive']=q.exists()
 if q.exists():d['child_status']=q.joinpath('status').read_text();d['child_cmdline']=q.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode();d['child_stat']=q.joinpath('stat').read_text()
print(json.dumps(d,sort_keys=True))
