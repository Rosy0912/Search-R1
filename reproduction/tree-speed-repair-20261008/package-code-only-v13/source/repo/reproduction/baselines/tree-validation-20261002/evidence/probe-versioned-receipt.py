import pathlib,tempfile,hashlib,json,time,subprocess,sys
root=pathlib.Path.cwd();p=root/'work/train_tree_stage_versioned.sh';s=p.read_text();code=s.split("<<'RECEIPT'\n",1)[1].split('\nRECEIPT',1)[0];results=[]
with tempfile.TemporaryDirectory() as td:
 d=pathlib.Path(td);logs=d/'logs';logs.mkdir();f=logs/'five.json';ev=d/'accepted-two-review';ev.write_text('immutable two review');out=d/'index.json';code=code.replace("'/dataset/shaoshitong/search-r1-rl/tree-grpo/logs'",repr(str(logs)))
 base={'permitted_stages':['five'],'index_deadline_epoch':time.time()+8000,'evidence_file_sha256':{str(ev):hashlib.sha256(ev.read_bytes()).hexdigest()},'index_instance':{'instance':'fixed GPU4 candidate'}}
 def run(label,record=base,stage='five',budget='7200',badsha=False,path=None):
  f.write_text(json.dumps(record));q=path or f
  if path:path.write_bytes(f.read_bytes())
  digest=hashlib.sha256(q.read_bytes()).hexdigest();digest='0'*64 if badsha else digest
  if out.exists():out.unlink()
  r=subprocess.run([sys.executable,'-c',code,str(q),digest,stage,str(out),budget],capture_output=True,text=True)
  good=label=='positive';assert (r.returncode==0)==good,(label,r.stderr);results.append({'case':label,'returncode':r.returncode,'accepted':r.returncode==0})
  if good:assert json.loads(out.read_text())==record['index_instance']
 run('positive');run('two-only',dict(base,permitted_stages=['two']));run('receipt-hash-changed',badsha=True);run('expired',dict(base,index_deadline_epoch=time.time()-1));run('insufficient-stage-time',dict(base,index_deadline_epoch=time.time()+7100));run('missing-budget',budget='');run('outside-own-logs',path=d/'outsider.json');ev.write_text('polluted');run('evidence-changed')
result={'scope':'actual launcher RECEIPT Python block extracted; no Slurm/GPU/production edits','cases':results,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()};(root/'work/aux-gate-review/versioned-receipt-fixtures.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
