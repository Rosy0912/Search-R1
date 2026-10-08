import pathlib,subprocess,shlex,json
B=pathlib.Path(__file__).resolve().parent.parent
code="""import pathlib,json,hashlib,base64,time,subprocess,gzip
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');run=root/'runs/bapo-instruct300-formal300-3292';names=['recovery-witness/driver.json']+['recovery-witness/rank_%d.json'%i for i in range(8)]+['resolved-config.yaml','permission-resolved-config.yaml','recovery-manifest.json','recovery-permission.json','recovery-permission-check.json','source-commit.txt','tools-manifest.json','train.log','phase.log','resource-watch.jsonl','retrieval-assets-locality.json','runtime-profile.txt','retriever-health.json'];out={}
for name in names:
 p=run/name;assert p.is_file()and not p.is_symlink()and p.resolve()==p;raw=p.read_bytes();assert len(raw)<32*1024*1024;out[str(p)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
for name in names[:9]:
 p=run/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==out[str(p)]['sha256']
slurm=subprocess.check_output(['scontrol','show','job','-dd','3292'],text=True,timeout=15)
result=json.dumps(dict(status='READ_ONLY_ACTUAL_FORMAL3292_RUNTIME_RESTORE_ORIGINALS_NOT_HARDWARE3287_OR_FINAL_ACCEPTANCE',job_id=3292,root=str(root),run_root=str(run),source_commit='a2568d0df306ead6e059760e28b5bd53ee3f6876',captured_at_unix=time.time(),files=out,slurm_live=slurm,signals_sent=0,additional_gpu_executed=False)).encode();print(json.dumps(dict(gzip_base64=base64.b64encode(gzip.compress(result)).decode(),bytes=len(result),sha256=hashlib.sha256(result).hexdigest())))
"""
cmd='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/env/bin/python -B -c '+shlex.quote(code)
r=subprocess.run(['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B',str(B/'h100_exec.py'),'--socks','--timeout','60',cmd],capture_output=True,timeout=75)
(B/'evidence/ray-envfix-formal3292-runtime-restore-compressed-transfer.json').write_bytes(r.stdout);(B/'evidence/ray-envfix-formal3292-runtime-restore-original-capture.stderr').write_bytes(r.stderr);print(json.dumps(dict(exit_code=r.returncode,bytes=len(r.stdout),stderr=r.stderr.decode())));assert r.returncode==0 and not r.stderr

import base64,gzip,hashlib
transfer=json.loads(r.stdout);raw=gzip.decompress(base64.b64decode(transfer['gzip_base64'],validate=True));assert len(raw)==transfer['bytes']and hashlib.sha256(raw).hexdigest()==transfer['sha256'];(B/'evidence/ray-envfix-formal3292-runtime-restore-original-capture.json').write_bytes(raw);print('Exact decompressedoriginal',len(raw),transfer['sha256'])
