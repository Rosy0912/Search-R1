import pathlib,json,hashlib,base64,sys
sys.path.insert(0,'work/g1-native-cpu-deploy-v6');from stream_transport import send
src=r'''
import pathlib,json,hashlib,base64,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-grpo100-common-v3-r2/evidence');out={'job_id':3218,'epoch':time.time(),'files':[]}
for p in sorted((root/'actor-resource-audit').glob('actor-updates-rank*-outer1.json'))+[root.parent/'actual-outer.json']:
 a=p.stat()
 with p.open('rb') as f:b=f.read(a.st_size)
 out['files'].append({'path':str(p),'observed_size':a.st_size,'sha256':hashlib.sha256(b).hexdigest(),'size':len(b),'base64':base64.b64encode(b).decode(),'running_snapshot_not_final':True})
p=root/'own-resource-samples.jsonl';a=p.stat()
with p.open('rb') as f:
 off=max(0,a.st_size-2*1024*1024);f.seek(off);b=f.read(a.st_size-off)
lines=b.splitlines(keepends=True);raw=[x for x in lines[1 if off else 0:] if x.endswith(b'\n')][-1];r=json.loads(raw)
out['resource']={'path':str(p),'observed_file_size':a.st_size,'row_size':len(raw),'row_sha256':hashlib.sha256(raw).hexdigest(),'row_base64':base64.b64encode(raw).decode(),'epoch':r['epoch'],'cgroup_memory':r['cgroup_memory'],'gpus':r['gpu']['gpus'],'stage_present':'stage' in r}
print(json.dumps(out))
'''
r=send(src,attempts=3,retry_safe=True,timeout=90);r.check_returncode();out=json.loads(r.stdout);d=pathlib.Path('outputs/remote-evidence/standard3218-outer1-originals-'+str(int(out['epoch']*1e6)));d.mkdir(exist_ok=False)
for f in out['files']:
 b=base64.b64decode(f.pop('base64'));assert hashlib.sha256(b).hexdigest()==f['sha256'];(d/pathlib.Path(f['path']).name).write_bytes(b)
b=base64.b64decode(out['resource'].pop('row_base64'));assert hashlib.sha256(b).hexdigest()==out['resource']['row_sha256'];(d/'resource-original-row.json').write_bytes(b);(d/'original-refs.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'directory':str(d),'refs':out}))
