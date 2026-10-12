import subprocess,pathlib,json,hashlib,time,datetime,zoneinfo
r=pathlib.Path('work/cleanup-h10011/source').resolve()
sha=subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','-C',str(r),'rev-parse','FETCH_HEAD'],text=True).strip();assert sha==remote
records=[]
for fn in subprocess.check_output(['git','-C',str(r),'ls-files','-z']).decode().split('\0'):
 if not fn:continue
 p=r/fn
 if p.is_file():b=p.read_bytes();records.append(dict(path=fn,size=len(b),sha256=hashlib.sha256(b).hexdigest()))
j=dict(epoch=time.time(),verified_shanghai=datetime.datetime.now(zoneinfo.ZoneInfo('Asia/Shanghai')).isoformat(),session='01a0fcbd-41d8-7910-a410-372ce5c8870c',runtime_source_commit='0fe642bd0c71f0b5f9813cbbad7bdd94e0625244',repo='https://github.com/shaoshitong/Search-R1',branch='codex/backup-bapo-cleanup-20261012',backup_commit=sha,fresh_remote_sha=remote,file_count=len(records),actual_runtime_loose_files=47,failed_runtime_loose_files=135,files=records)
pathlib.Path('outputs/cleanup-h10011/source-receipt.json').write_text(json.dumps(j,indent=2))
print(sha,len(records),j['verified_shanghai'])
