import subprocess,json,base64,pathlib,hashlib
ROOT='/dataset/shaoshitong/search-r1-rl/topd';paths=[ROOT+'/logs/standard-search-r1-teacher-identity-v3-native-cpu-submission.json',ROOT+'/logs/standard-search-r1-teacher-identity-v3-native-cpu-3160-actual-final.json',ROOT+'/logs/standard-search-r1-teacher-identity-v3-native-cpu-3160-control.jsonl',ROOT+'/logs/standard-r1-teacher-identity-v3-cpu-3160.log']+[ROOT+'/logs/standard-r1-teacher-identity-v3-cpu-3160/'+n for n in ('result.json','chain.log','score.log')]
script="python3 - <<'CPU_READONLY_RAW'\nimport pathlib,hashlib,base64,json\npaths="+repr(paths)+"\nfor name in paths:\n p=pathlib.Path(name);b=p.read_bytes();print(json.dumps({'path':name,'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()}))\nCPU_READONLY_RAW\n"
r=subprocess.run(['python3','work/remote_topd.py'],input=script,text=True,capture_output=True);assert r.returncode==0,r.stderr
out=pathlib.Path('outputs/remote-evidence/standard-identity-native3160');out.mkdir(exist_ok=True)
for line in r.stdout.splitlines():
 v=json.loads(line);b=base64.b64decode(v.pop('data'));assert hashlib.sha256(b).hexdigest()==v['sha256'];p=out/pathlib.Path(v['path']).name;assert not p.exists();p.write_bytes(b);print(v)
