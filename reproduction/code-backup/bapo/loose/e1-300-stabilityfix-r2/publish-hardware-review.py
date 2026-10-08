import pathlib,sys,json,base64,hashlib,subprocess
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
x=json.loads(sys.stdin.buffer.read());receipt=base64.b64decode(x['receipt_base64'],validate=True);report=base64.b64decode(x['report_base64'],validate=True);r=json.loads(receipt)
assert r['status']=='PASS_INDEPENDENT_E1_HARDWARE_REVIEW'and r['source_commit']=='39cdcdd1404213b897bb0de99cc52d72707682f0'and r['root']==str(ROOT)
assert type(r['job_id'])is int and r['job_id']==3220 and r['run_root']==str(ROOT/'runs/bapo-instruct300-hardware-3220')
assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==r['source_commit']
assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True)
assert 'JobState=COMPLETED'in(ROOT/'runs/bapo-instruct300-hardware-3220/slurm-terminal.txt').read_text()
p=pathlib.Path(r['independent_review']['path']);assert p.parent==ROOT/'logs'and p.parent.resolve()==p.parent and not p.exists()and not p.is_symlink();assert hashlib.sha256(report).hexdigest()==r['independent_review']['sha256']
q=ROOT/'logs/hardware-acceptance.json';assert not q.exists()and not q.is_symlink()
p.open('xb').write(report);q.open('xb').write(receipt)
print(json.dumps({'status':'FRESH_INDEPENDENT_HARDWARE_REVIEW_DEPLOYED_REQUIRES_ACTUAL_GATE','receipt_sha256':hashlib.sha256(receipt).hexdigest(),'report_sha256':hashlib.sha256(report).hexdigest(),'job_id':3220}))
