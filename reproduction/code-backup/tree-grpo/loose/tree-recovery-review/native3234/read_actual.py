import pathlib,json,hashlib,subprocess,time,socket
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-recovery300-3234';tools=root/'instruct300/candidates/recovery-502e9fa-f91c4f17'
paths=[run/'allocation.txt',run/'launch-manifest.json',run/'native-fullstate.log',run/'gpu-kl.log',run/'run-result.json',run/'output/recovery-loaded.json',run/'resource-samples.jsonl']
paths += [run/'native-fullstate'/f'rank-{i:02d}.json' for i in range(8)]
paths += [run/'native-fullstate'/f'rank-{i:02d}-diagnostic.json' for i in range(8)]
paths += [run/'gpu-kl-proof'/f'rank-{i}.json' for i in range(8)]
paths += [tools/'manifest.json',tools/'PRE_DISPATCH_RECEIPT.json',tools/'native_fullstate_probe.py',tools/'gpu-kl-proof.py',run/'resolved.yaml']
files=[];missing=[]
for p in paths:
 if not p.is_file():missing.append(str(p));continue
 raw=p.read_bytes();files.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode()))
scheduler=subprocess.run(['scontrol','show','job','3234'],capture_output=True,text=True)
print(json.dumps(dict(snapshot_epoch=time.time(),hostname=socket.gethostname(),job_id='3234',run=str(run),files=files,missing=missing,scontrol_stdout=scheduler.stdout,scontrol_stderr=scheduler.stderr,scontrol_returncode=scheduler.returncode)),flush=True)
