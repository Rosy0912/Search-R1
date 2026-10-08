import subprocess,sys
job=sys.argv[1]; assert job.isdigit()
script='''from pathlib import Path
import json
r=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/runs/opsd-capacity-'''+job+'''/evidence')
for name in ('micro-capacity.json','micro-1.json','micro-2.json','micro-4.json','micro-8.json'):
 p=r/name
 if not p.exists(): continue
 v=json.loads(p.read_text())
 if 'ranks' in v:
  rows=v.pop('ranks')
  v['all_eight_passed']=len(rows)==8 and all(x['passed'] for x in rows)
  v['max_estimated_device_used_gib']=max(x['estimated_peak_device_used_bytes'] for x in rows)/2**30
  v['max_reserved_gib']=max(x['peak_reserved_bytes'] for x in rows)/2**30
  v['training_flags']=rows[0].get('student_checkpoint_training_runtime')
  v['checkpoint_keys']=[k for k in rows[0] if 'decoder' in k]
 else:
  for c in v.get('candidates',[]): c.pop('before_idle',None)
  v.pop('after_idle',None)
 print(name,json.dumps(v))
'''
command="/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python - <<'PYREMOTE'\n"+script+"\nPYREMOTE"
raise SystemExit(subprocess.run([sys.executable,'work/remote.py',command]).returncode)
