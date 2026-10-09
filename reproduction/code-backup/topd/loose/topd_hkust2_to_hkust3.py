"""Copy an accepted Wiki archive directly from HKUST2 to HKUST3, then SHA verify."""
import fcntl, json, os, pathlib, shlex, subprocess, time
root=pathlib.Path('/data/shaoshitong/topd-hfd-20261009')
lock=(root/'wiki-forward.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=json.loads((root/'wiki_corpus-download-verified.json').read_text())
assert receipt['status']=='SOURCE_HF_DOWNLOAD_ARCHIVE_SHA256_VERIFIED'
source=root/'official/topd-migration-20261009/archives/wiki_corpus.tar.zst'
dest='/backup/shaoshitong/topd-hf-migration-20261009/hkust2-forward/wiki_corpus.tar.zst'
ssh=['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','shaoshitong@10.120.17.122']
(root/'wiki-forward-controller.json').write_text(json.dumps({'pid':os.getpid(),'epoch':time.time(),'route':'HKUST2->HKUST3','mac_used':False,'destination':dest}))
subprocess.run(['rclone','copyto',str(source),':sftp:'+dest,'--sftp-ssh',shlex.join(ssh),'--transfers','1','--checkers','1','--stats','15s','--stats-one-line','--log-level','INFO'],check=True)
code="import pathlib,json,hashlib,time;p=pathlib.Path("+repr(dest)+");h=hashlib.sha256();f=p.open('rb');\nfor b in iter(lambda:f.read(4194304),b''):h.update(b)\nassert p.stat().st_size==5564130143 and h.hexdigest()=='63756238c05cf00f5531795b965677cbea8b3cd5be6f29ff3e4b431401b24d74';r={'group':'wiki_corpus','bytes':p.stat().st_size,'sha256':h.hexdigest(),'epoch':time.time(),'status':'HKUST2_HKUST3_ARCHIVE_SHA256_VERIFIED','mac_used':False};p.with_name('wiki_corpus-forward-verified.json').write_text(json.dumps(r));print(json.dumps(r))"
subprocess.run(ssh+['python3 -c '+shlex.quote(code)],check=True)
