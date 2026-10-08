import base64,hashlib,json,subprocess,sys
from pathlib import Path
root='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004'
helper=Path('work/gigpo-e1-quality-resume-runtime/safety-calibration-queue-dispatch-v9/submit_queued_v9_once_v1.py')
review=Path(sys.argv[1]); r=json.loads(review.read_text())
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='23e1e25d60689f9d8730dfd0e4f21167a7007498bf8f608753e08cae7a818950'
assert r['status']=='PASS' and r['binding_sha256']=='7cc3098a35d6f56e809e293e2f26642cc365eac8c49c1ddaedb592e989c79ce2' and r['submit_helper_sha256']==hashlib.sha256(helper.read_bytes()).hexdigest()
items=[]
for f,destination in [(helper,root+'/tools/policy-safety-calibration-queue-dispatch-v9/'+helper.name),(review,root+'/logs/'+review.name)]:
 raw=f.read_bytes();items.append({'path':destination,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'b64':base64.b64encode(raw).decode()})
code="""import base64,hashlib,json,os,socket,datetime
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-8'
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
assert hashlib.sha256((root/'logs/policy-safety-calibration-52befe9-reviewed-binding-v9.json').read_bytes()).hexdigest()=='7cc3098a35d6f56e809e293e2f26642cc365eac8c49c1ddaedb592e989c79ce2'
items=ITEMS
folder=root/'tools/policy-safety-calibration-queue-dispatch-v9'
intent=root/'logs/policy-safety-queued-dispatch-v9-install-intent.json'
receipt=root/'logs/policy-safety-queued-dispatch-v9-install-receipt.json'
for f in [folder,intent,receipt,root/'logs/policy-safety-calibration-52befe9-submit-intent-v9.json',root/'logs/policy-safety-calibration-52befe9-submit-receipt-v9.json',root/'logs/policy-safety-calibration-52befe9-queue-submit-intent-v9.json',root/'logs/policy-safety-calibration-52befe9-queue-submit-receipt-v9.json']:
 assert not os.path.lexists(f),'existing claim/path: inspect preserve'
for row in items:
 assert not os.path.lexists(row['path']),'destination exists preserve'
 raw=base64.b64decode(row['b64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
def put(path,data):
 fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o644)
 with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 fd=os.open(Path(path).parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
plain=[{k:v for k,v in row.items() if k!='b64'} for row in items]
put(intent,(json.dumps({'status':'CLAIMED_INSTALL_NO_RETRY','files':plain},indent=2)+chr(10)).encode())
installed=[]
try:
 os.mkdir(folder)
 for row in items:
  put(row['path'],base64.b64decode(row['b64']));assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'];installed.append(row['path'])
 result={'status':'EXACT_EXTERNAL_QUEUED_DISPATCHER_INSTALLED_NOT_SUBMITTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':plain,'installed':installed,'frozen_v9_changed':False,'gpu_used':False,'submitted':False}
except BaseException as error:
 result={'status':'PARTIAL_INSTALL_PRESERVE_NO_RETRY','installed':installed,'error':str(error)}
 put(receipt,(json.dumps(result,indent=2)+chr(10)).encode());raise
put(receipt,(json.dumps(result,indent=2)+chr(10)).encode())
raw=receipt.read_bytes();print(json.dumps({'receipt_path':str(receipt),'receipt_sha256':hashlib.sha256(raw).hexdigest(),'receipt_raw':raw.decode(),'result':result}))
""".replace('ITEMS',repr(items))
compile(code,'remote-install','exec')
script="/usr/bin/python3 - <<'REMOTE'\n"+code+"\nREMOTE\n"
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=160)
result={'transport_returncode':p.returncode,'stderr':p.stderr,'raw_stdout':p.stdout}
if p.returncode==0:result['actual']=json.loads(p.stdout)
out=Path('outputs/gigpo-e1-policy-safety/CALIBRATION-QUEUED-DISPATCH-ACTUAL-INSTALL-v9.json')
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'returncode':p.returncode,'actual':result.get('actual')}))
