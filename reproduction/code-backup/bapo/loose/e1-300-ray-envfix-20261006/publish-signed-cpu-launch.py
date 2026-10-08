import pathlib,json,base64,hashlib,subprocess,os,sys
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');source='a2568d0df306ead6e059760e28b5bd53ee3f6876'
assert r==r.resolve() and all((r/n).is_dir() and not (r/n).is_symlink() and (r/n).resolve()==r/n for n in ['logs','repo','runs']) and not list((r/'runs').iterdir())
assert (r/'source-expected.txt').read_text()==source+'\n' and subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True)
p=json.load(sys.stdin)['files'];assert set(p)=={'final-launch-independent-review.md','cpu-gate.json','cpu-gate-source.txt','launch-approval.json'}
raws={n:base64.b64decode(v['base64'],validate=True) for n,v in p.items()}
for n,raw in raws.items():assert len(raw)==p[n]['bytes'] and hashlib.sha256(raw).hexdigest()==p[n]['sha256']
a=json.loads(raws['launch-approval.json']);assert a['status']=='PASS_INDEPENDENT_E1_LAUNCH_REVIEW' and a['source_commit']==source and a['root']==str(r) and a['actual_gpu']is False and a['production_training_permission_signed']is False
assert hashlib.sha256(raws['launch-approval.json']).hexdigest()=='400aedda99048c3c16229f26daa7a521b628fe83f7a774ebd981d5d435c6f0b2' and hashlib.sha256((r/'logs/tools-manifest.json').read_bytes()).hexdigest()==a['tools_manifest_sha256']
for n in ['final-launch-independent-review.md','cpu-gate.json','cpu-gate-source.txt','launch-approval.json']:
 with(r/'logs'/n).open('xb') as f:f.write(raws[n])
subprocess.run([str(r/'env/bin/python'),'-B',str(r/'tools/verify_launch.py'),'hardware'],cwd=r/'repo',env=dict(os.environ,PYTHONPATH=str(r/'repo'),SPT_NOENV='1'),check=True)
print('R3_ACTUAL_LAUNCH_GATE_PASS_NO_JOB_SUBMITTED')
