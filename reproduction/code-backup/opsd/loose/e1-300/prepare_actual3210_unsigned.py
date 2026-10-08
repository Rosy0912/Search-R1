import pathlib,json,hashlib,shlex,datetime,subprocess
E=pathlib.Path(__file__).resolve().parent
ns={'__file__':str(E/'deploy_formal_reviewed.py')};s=pathlib.Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns)
R='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004';run=R+'/runs/formal-3210';repo=R+'/repo-a0968c2d24be';tools=R+'/tools/first5-e4d8bd9e088e'
pins={'spec.json':('actual3210-final5role-evidence-spec-root.json','d5b585feedcec814eef8416c073e85b636b621a190a210c0f68476c1c58e749f'),'root-review.json':('ACTUAL3210_FIRST5_ROOT_REVIEW_RECORD.json','bd6ecd338d698544805a915280588f8ca0b8b6eb9f8ba46fde13e13ea6c4f21d'),'independent-review.json':('ACTUAL3210_FIRST5_INDEPENDENT_REVIEWER_ACCEPTED_RECORD.json','060415b1da78999a1d13be1007773dce6a9988750f46e7196755bdca5421208c')}
inputs={}
for name,(local,pin) in pins.items():
 b=(E/local).read_bytes();assert hashlib.sha256(b).hexdigest()==pin;inputs[name]=b.decode()
D=run+'/evidence/actual-first5-unsigned-review-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
cmd=['srun','--jobid=3210','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','--nodelist=h100-10','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+repo,R.replace('/opsd-e1-instruct300-20261004','/opsd')+'/env/bin/python',tools+'/receipt_payload.py','--run-root',run,'--repo',repo,'--manifest',run+'/full_state/global_step_5/manifest.json','--manifest-sha256','44b94a6605568ebf8fa745a41a7f1df3444deb1fec187c2a7c2cbb619643dc7b','--evidence-spec',D+'/spec.json','--evidence-spec-sha256',pins['spec.json'][1],'--root-review',D+'/root-review.json','--root-review-sha256',pins['root-review.json'][1],'--independent-review',D+'/independent-review.json','--independent-review-sha256',pins['independent-review.json'][1],'--tools-manifest',tools+'/manifest.json','--tools-manifest-sha256','e4d8bd9e088e49fc92f335abfaa47737830d1ba9320cbe206d1bd130876518f7','--output',D+'/unsigned-step5-payload.json']
controller='''import subprocess,pathlib,json,datetime,os,socket
D=pathlib.Path(%r);cmd=%r
with (D/'producer-stdout-original.txt').open('xb') as out,(D/'producer-stderr-original.txt').open('xb') as err:
 result=subprocess.run(cmd,stdout=out,stderr=err)
with (D/'producer-exit-original.json').open('x') as f:json.dump({'exit_code':result.returncode,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_pid':os.getpid(),'node':socket.gethostname(),'command':cmd},f,sort_keys=True,indent=2)
'''%(D,cmd)
code='''import pathlib,json,hashlib,os,socket,subprocess
D=pathlib.Path(%r);assert os.getuid()==0 and socket.gethostname()=='h100-10';assert D.parent.resolve(strict=True)==D.parent
D.mkdir(mode=0o700)
inputs=%r;pins=%r
for name,text in inputs.items():
 b=text.encode();assert hashlib.sha256(b).hexdigest()==pins[name][1]
 with (D/name).open('xb') as f:f.write(b)
 os.chmod(D/name,0o400)
with (D/'producer-command.json').open('x') as f:json.dump(%r,f,indent=2)
with (D/'controller-original.txt').open('xb') as out:
 p=subprocess.Popen(['python3','-c',%r],stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True)
with (D/'launch-original.json').open('x') as f:json.dump({'directory':str(D),'controller_pid':p.pid,'scope':'Unsigned preparation only; no signing or release'},f,sort_keys=True)
print(json.dumps({'directory':str(D),'controller_pid':p.pid}))
'''%(D,inputs,pins,cmd,controller)
(E/'actual3210-unsigned-launch-intent.json').write_text(json.dumps({'directory':D,'command':cmd,'pins':pins},indent=2))
r=subprocess.run(ns['route']('python3 -c '+shlex.quote(code)),capture_output=True,timeout=60)
(E/'actual3210-unsigned-launch-stdout-original.txt').write_bytes(r.stdout);(E/'actual3210-unsigned-launch-stderr-original.txt').write_bytes(r.stderr)
print(r.returncode,r.stdout.decode(),r.stderr.decode());assert r.returncode==0
