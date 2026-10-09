"""HKUST2 accepted missing archives -> HKUST3 -> HPC3, no Mac data path."""
import fcntl,hashlib,json,os,pathlib,shlex,subprocess,time
root=pathlib.Path(__file__).resolve().parent
lock=(root/'missing-forward.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((root/'plan.json').read_text())
ssh=['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','shaoshitong@10.120.17.122']
def save(n,d):
 p=root/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(d));os.replace(t,p)
save('missing-forward-controller.json',{'pid':os.getpid(),'epoch':time.time(),'mac_used':False,'groups':[x['group'] for x in plan['records']]})
for record in plan['records']:
 name=record['group'];source=root/(name+'.tar.zst');receipt=root/(name+'-download-verified.json')
 while not (source.exists() and receipt.exists()):time.sleep(15)
 accepted=json.loads(receipt.read_text());assert accepted['sha256']==record['archive_sha256'] and accepted['bytes']==record['archive_bytes']
 dest='/backup/shaoshitong/topd-hf-migration-20261009/hkust2-forward/'+name+'.tar.zst'
 failures=0
 while True:
  try:
   subprocess.run(['rclone','copyto',str(source),':sftp:'+dest,'--sftp-ssh',shlex.join(ssh),'--transfers','1','--checkers','1','--stats','15s','--stats-one-line','--log-level','INFO'],check=True)
   code="import pathlib,json,hashlib,time;p=pathlib.Path("+repr(dest)+");h=hashlib.sha256();f=p.open('rb');\nfor b in iter(lambda:f.read(4194304),b''):h.update(b)\nassert p.stat().st_size=="+str(record['archive_bytes'])+" and h.hexdigest()=="+repr(record['archive_sha256'])+";r={'group':"+repr(name)+",'bytes':p.stat().st_size,'sha256':h.hexdigest(),'epoch':time.time(),'status':'HKUST2_HKUST3_ARCHIVE_SHA256_VERIFIED','mac_used':False};p.with_name("+repr(name+'-forward-verified.json')+").write_text(json.dumps(r));print(json.dumps(r))"
   subprocess.run(ssh+['python3 -c '+shlex.quote(code)],check=True)
   remote_root='/backup/shaoshitong/topd-hf-migration-20261009/direct-hpc3'
   subprocess.run(ssh+['python3 '+remote_root+'/topd_remote_missing_target.py '+shlex.quote(name)],check=True)
   save(name+'-remote-chain-verified.json',{'group':name,'epoch':time.time(),'status':'REMOTE_CHAIN_ARCHIVE_AND_ALL_FILES_SHA_VERIFIED','mac_used':False})
   break
  except Exception as e:
   failures+=1;save(name+'-forward-failure.json',{'epoch':time.time(),'count':failures,'error_type':type(e).__name__});time.sleep(min(180,30*failures))
save('remote-chain-complete.json',{'epoch':time.time(),'status':'ALL_MISSING_GROUPS_REMOTE_CHAIN_VERIFIED','mac_used':False,'note':'Global fresh SHA and actual targetcomplete still required'})
