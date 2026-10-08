import pathlib,json,hashlib,base64
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/scheduler-head');files={}
for name in ['hardware-job-id.txt','hardware-submit-intent.txt','submit.sh','hardware.sbatch','approved-launch-hashes.json','hardware-3287-terminal.txt']:
 p=root/name;assert p.is_file() and not p.is_symlink();a=p.stat();raw=p.read_bytes();b=p.stat();assert (a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns);files[name]={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
print(json.dumps({'files':files},sort_keys=True))
