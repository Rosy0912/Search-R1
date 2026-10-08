"""Only run after exact publisher code approval and final signed-ready receipt."""
import pathlib,json,hashlib,base64,subprocess,shlex,sys
D=pathlib.Path(__file__).resolve().parent;B=D.parent;ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006'
def row(path,raw):return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
receipt=(D/'hardware3287-final-receipt.json').read_bytes();r=json.loads(receipt);assert r['publication_ready']is True
report=(D/'HARDWARE3287-FINAL-INDEPENDENT-REVIEW.md').read_bytes();assert hashlib.sha256(report).hexdigest()==r['independent_review']['sha256']
payload=dict(receipt=row(ROOT+'/logs/hardware-acceptance.json',receipt),report=row(r['independent_review']['path'],report));payloadp=D/'signed-hardware3287-publication-payload.json';payloadp.write_text(json.dumps(payload)+'\n')
code=(D/'publish-reviewed-hardware3287.py').read_bytes();approval=json.loads((D/'hardware3287-publisher-independent-code-approval.json').read_text());assert approval['publisher_sha256']==hashlib.sha256(code).hexdigest()
command=ROOT+'/env/bin/python -B -c '+shlex.quote('import base64;exec(compile(base64.b64decode('+repr(base64.b64encode(code).decode())+'),"reviewed-hardware-publication","exec"))')
result=subprocess.run(['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B',str(B/'h100_exec.py'),'--socks','--timeout','600','--stdin-file',str(payloadp),command],capture_output=True,timeout=615)
(B/'evidence/ray-envfix-hardware3287-final-publication-gate.stdout').write_bytes(result.stdout);(B/'evidence/ray-envfix-hardware3287-final-publication-gate.stderr').write_bytes(result.stderr);print(json.dumps(dict(ssh_exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode())));assert result.returncode==0
