"""Forward only accepted CoSearch supplement HKUST2 -> HKUST3; no token."""
import fcntl, hashlib, json, os, pathlib, shlex, subprocess, time
ROOT = pathlib.Path(__file__).resolve().parent
DEST = '/backup/shaoshitong/cosearch-hf-migration-20261009'
SSH = ['ssh', '-T', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes', '-oConnectTimeout=8', 'shaoshitong@10.120.17.122']
def save(name, data):
    p=ROOT/name; t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));os.replace(t,p)
def main():
    lock=(ROOT/'forward.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save('forward-controller.json',{'pid':os.getpid(),'epoch':time.time(),'route':'HKUST2 -> HKUST3','mac_data_relay':False})
    record=json.loads((ROOT/'plan.json').read_text())['records'][0]
    receipt=ROOT/(record['group']+'-download-verified.json')
    while not receipt.exists():
        save('forward-progress.json',{'epoch':time.time(),'status':'WAITING_DOWNLOAD_SHA'})
        time.sleep(15)
    accepted=json.loads(receipt.read_text());assert accepted['sha256']==record['archive_sha256']
    source=ROOT/(record['group']+'.tar.zst');assert source.stat().st_size==record['archive_bytes']
    # Authentication and current writable capacity are checked on the actual hop.
    probe="import pathlib,os,json; p=pathlib.Path(%r);p.mkdir(parents=True,exist_ok=True);v=os.statvfs(p);assert v.f_bavail*v.f_frsize>%d; q=p/'.write-probe';q.write_bytes(b'cosearch');q.unlink();print('AUTH_WRITE_SPACE_OK')"%(DEST,record['archive_bytes']*2)
    attempt=0
    while True:
        attempt+=1
        try:
            subprocess.run(SSH+['python3 -c '+shlex.quote(probe)],check=True,timeout=30,capture_output=True)
            destination=DEST+'/'+source.name
            save('forward-progress.json',{'epoch':time.time(),'status':'FORWARDING','attempt':attempt})
            subprocess.run(['rclone','copyto',str(source),':sftp:'+destination,'--sftp-ssh',shlex.join(SSH),'--transfers','1','--checkers','1','--stats','30s','--log-level','ERROR'],check=True,timeout=3600)
            code="import pathlib,hashlib,json,time;p=pathlib.Path(%r);h=hashlib.sha256();f=p.open('rb');\nfor b in iter(lambda:f.read(4194304),b''):h.update(b)\nassert p.stat().st_size==%d and h.hexdigest()==%r;r={'group':%r,'bytes':p.stat().st_size,'sha256':h.hexdigest(),'epoch':time.time(),'status':'HKUST2_HKUST3_ARCHIVE_SHA256_VERIFIED','mac_data_relay':False};p.with_name('forward-verified.json').write_text(json.dumps(r));print(json.dumps(r))"%(destination,record['archive_bytes'],record['archive_sha256'],record['group'])
            result=subprocess.run(SSH+['python3 -c '+shlex.quote(code)],check=True,capture_output=True,text=True,timeout=600)
            save('forward-verified.json',json.loads(result.stdout));break
        except subprocess.CalledProcessError as error:
            # A failed SHA check is an integrity stop, never a successful hop.
            if 'AssertionError' in str(error.stderr):
                raise
            save('forward-failure.json',{'epoch':time.time(),'attempt':attempt,'error_type':type(error).__name__})
            time.sleep(min(120,15*attempt))
        except subprocess.TimeoutExpired:
            save('forward-failure.json',{'epoch':time.time(),'attempt':attempt,'error_type':'TimeoutExpired'})
            time.sleep(min(120,15*attempt))
if __name__=='__main__':
    main()
