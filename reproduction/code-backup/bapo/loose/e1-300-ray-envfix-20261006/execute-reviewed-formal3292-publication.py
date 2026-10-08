import pathlib,hashlib,json,base64,subprocess,shlex
D=pathlib.Path(__file__).resolve().parent;B=D.parent
report=D/'FORMAL3292-RESUME25-INDEPENDENT-PERMISSION-REVIEW.md';permission=D/'formal3292-resume25-permission.json'
raw=permission.read_bytes();p=json.loads(raw);assert p['status']=='PASS_INDEPENDENT_BAPO_RESUME25_REVIEW'and p['target_job_id']==3292;assert p['independent_review']['sha256']==hashlib.sha256(report.read_bytes()).hexdigest()
files={x.name:dict(text=x.read_text(),sha256=hashlib.sha256(x.read_bytes()).hexdigest())for x in [report,permission]};files['recovery-permission.json']=files.pop(permission.name);payload=D/'formal3292-signed-publication-payload.json';payload.write_text(json.dumps(dict(job_id=3292,files=files))+'\n')
code=(D/'publish-formal-permission.py').read_bytes();assert hashlib.sha256(code).hexdigest()=='74a489599124e537de24066178fca9a5f48ccdb2d5d17d6113a586f6a8955679'
remote="""import os,sys,pathlib,base64,subprocess,json,hashlib
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');os.environ['PYTHONPATH']=str(root/'repo');os.environ['PYTHONDONTWRITEBYTECODE']='1';os.chdir(root/'repo')
exec(compile(base64.b64decode(__PUBLISHER_BASE64__),'reviewed-formal-publisher','exec'))
gate=root/'tools/check-recovery-permission.py';manifest=json.loads((root/'logs/tools-manifest.json').read_text());assert hashlib.sha256(gate.read_bytes()).hexdigest()==manifest['files'][gate.name]
r=subprocess.run([str(root/'env/bin/python'),'-B',str(gate),'--run',str(root/'runs/bapo-instruct300-formal300-3292'),'--phase','formal300'],capture_output=True,text=True)
record=dict(status='ACTUAL_FROZEN_FORMAL3292_PERMISSION_GATE_EXECUTION',job_id=3292,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),additional_job_submitted=False)
p=root/'logs/formal3292-actual-permission-gate-execution.json'
with p.open('x')as f:f.write(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record));raise SystemExit(r.returncode)
""".replace('__PUBLISHER_BASE64__',repr(base64.b64encode(code).decode()))
compile(remote,'actual-remote-wrapper','exec')
cmd='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/env/bin/python -B -c '+shlex.quote(remote)+' --hardware-receipt-sha e8625dabce8c9cf5a97498cb0989b399937fd9719acfa186357b1d29cebe1e2c'
r=subprocess.run(['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B',str(B/'h100_exec.py'),'--socks','--timeout','600','--stdin-file',str(payload),cmd],capture_output=True,timeout=615)
(B/'evidence/ray-envfix-formal3292-permission-publication.stdout').write_bytes(r.stdout);(B/'evidence/ray-envfix-formal3292-permission-publication.stderr').write_bytes(r.stderr);print(json.dumps(dict(ssh_exit_code=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode())));assert r.returncode==0 and not r.stderr
