import base64, hashlib, json, pathlib, shlex, subprocess
base=pathlib.Path(__file__).resolve().parent
script=base/'tool_assets_full_verify.py'; manifest=base.parent/'opsd-e1-300/reproduction/h20/assets.json'
sha=hashlib.sha256(script.read_bytes()).hexdigest(); root='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/tools/tool-assets-full-verify-04f8-'+sha[:12]
wrapper='''#!/bin/bash
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
/usr/bin/python3 tool_assets_full_verify.py --manifest assets-original-04f8.json --output evidence
result=$?
printf '%s\\n' "$result" > exit-code.tmp
mv -n exit-code.tmp exit-code.txt
exit "$result"
'''
files={'tool_assets_full_verify.py':script.read_bytes(),'assets-original-04f8.json':manifest.read_bytes(),'run-cpu.sh':wrapper.encode()}
command='umask 077; mkdir '+shlex.quote(root)+' || exit 91; '
for name,data in files.items():
    command+='printf %s '+shlex.quote(base64.b64encode(data).decode())+' | base64 -d > '+shlex.quote(root+'/'+name)+'; '
command+='cd '+shlex.quote(root)+'; sha256sum tool_assets_full_verify.py assets-original-04f8.json; nohup bash run-cpu.sh > raw-stdout.log 2>&1 < /dev/null & pid=$!; printf "%s\\n" "$pid" > wrapper-pid.txt; cat wrapper-pid.txt'
r=subprocess.run(['python3',str(base.parent/'remote.py'),command],capture_output=True,text=True)
(base/'tool-assets-verifier-deployment.txt').write_text(r.stdout+r.stderr)
(base/'tool-assets-verifier-deployment.json').write_text(json.dumps({'remote_dir':root,'script_sha256':sha,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'deployment_exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
print(r.stdout,r.stderr); raise SystemExit(r.returncode)
