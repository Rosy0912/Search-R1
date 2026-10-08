import os,pathlib,subprocess,json,hashlib,time,re,base64
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004');repo=root/'repo';source=(root/'source-expected.txt').read_text().strip()
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
script=root/'logs/cpu-tests-fixed.py';data=base64.b64decode('IiIiUnVuIHJlcG9zaXRvcnkgdGVzdHMgd2l0aG91dCBpbnN0YWxsZWQgcGFja2FnZXMgc2hhZG93aW5nIGl0cyBuYW1lc3BhY2UuIiIiCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAppbXBvcnQgc3lzCmltcG9ydCBvcwppbXBvcnQgdHlwZXMKaW1wb3J0IHB5dGVzdApyb290ID0gUGF0aC5jd2QoKQojIENQVSB0ZXN0cyBhc3NlcnQgZGVmYXVsdCBjb21wb3NpdGlvbiBhbmQgdXNlIGxvY2FsIGxvb3BiYWNrIGZvciBHbG9vIHJhbmtzLgpvcy5lbnZpcm9uLnBvcCgiU0VBUkNIX1IxX0FTU0VUUyIsIE5vbmUpCm9zLmVudmlyb25bIkdMT09fU09DS0VUX0lGTkFNRSJdID0gImxvMCIgaWYgc3lzLnBsYXRmb3JtID09ICJkYXJ3aW4iIGVsc2UgImxvIgpmb3IgbmFtZSwgZGlyZWN0b3J5IGluIFsoJ3Rlc3RzJywgcm9vdC8ndGVzdHMnKSwgKCd0ZXN0cy5iYXNlbGluZXMnLCByb290Lyd0ZXN0cy9iYXNlbGluZXMnKV06CiAgICBtb2R1bGUgPSB0eXBlcy5Nb2R1bGVUeXBlKG5hbWUpCiAgICBtb2R1bGUuX19wYXRoX18gPSBbc3RyKGRpcmVjdG9yeSldCiAgICBzeXMubW9kdWxlc1tuYW1lXSA9IG1vZHVsZQppZiBfX25hbWVfXyA9PSAnX19tYWluX18nOgogICAgcmFpc2UgU3lzdGVtRXhpdChweXRlc3QubWFpbihbJy1xJywgJ3Rlc3RzL2Jhc2VsaW5lcycsICd0ZXN0cy9yZXByb2R1Y3Rpb24nXSkpCg==')
if script.exists():assert script.read_bytes()==data
else:script.write_bytes(data)
log=root/'logs/cpu-gate.log';assert not log.exists(),'Prior CPU gate exists: inspect without overwrite'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
started=time.monotonic()
with log.open('xb')as output:r=subprocess.run([str(root/'env/bin/python'),str(script)],cwd=repo,env=env,stdout=output,stderr=subprocess.STDOUT)
text=log.read_text();passed=re.findall(r'(\d+) passed',text);skipped=re.findall(r'(\d+) skipped',text);failed=re.findall(r'(\d+) failed',text)
record={'source_commit':source,'exit_code':r.returncode,'passed':int(passed[-1])if passed else 0,'skipped':int(skipped[-1])if skipped else 0,'failed':int(failed[-1])if failed else 0,'wall_seconds':time.monotonic()-started,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'test_launcher_sha256':hashlib.sha256(data).hexdigest(),'sys_prefix':str(root/'env'),'scope':'ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE'}
(root/'logs/cpu-gate.json').write_text(json.dumps(record,indent=2)+'\n');assert r.returncode==0 and record['passed']>=230 and record['failed']==0,record
(root/'logs/cpu-gate-source.txt').write_text(source+'\n');print(json.dumps(record),flush=True)
