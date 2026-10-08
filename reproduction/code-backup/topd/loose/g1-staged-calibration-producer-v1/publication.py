"""No head/manifest publication from incomplete original dump stages."""
import hashlib,json,os,pathlib
TRAIN_SHA='f12881dfbd2df07f51632949d06fa9dca5a41ad08ad9da1d7de5b9bd1e9117a0'
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def complete_dump(directory):
    directory=pathlib.Path(directory);r=json.loads((directory/'result.json').read_text())
    if not r.get('complete') or r.get('status')!='DUMP_COMPLETE_HARDWARE_NOT_INDEPENDENTLY_ACCEPTED':raise ValueError('no complete original dump')
    for phase in ('A','B'):
        barrier=json.loads((directory/f'phase-{phase}-owner-barrier.json').read_text())
        if barrier.get('returncode')!=0 or barrier.get('own_processes_remaining')!=[] or barrier.get('own_gpu_records_remaining')!=[] or not barrier.get('baseline_preserved'):raise ValueError('actual stage release/baseline failed')
        if barrier['config_sha256']!=r['config_sha256']:raise ValueError('stage config identity drift')
    config=json.loads((directory/'frozen-input-config.json').read_text())
    if sha(directory/'frozen-input-config.json')!=r['config_sha256']:raise ValueError('config bytes changed')
    dump=directory/'dump'/f"{config['dump_args']['tag']}_{config['dump_args']['offset']}.npz"
    if sha(dump)!=r['npz_sha256'] or sha(dump.with_suffix('.jsonl'))!=r['jsonl_sha256']:raise ValueError('dump output bytes changed')
    return config,r,dump

def publish(directory,manifest,heads,head_exit):
    config,result,dump=complete_dump(directory)
    if head_exit['returncode']!=0 or head_exit['own_processes_remaining']!=[] or head_exit['own_gpu_records_remaining']!=[] or not head_exit['baseline_preserved']:raise ValueError('original heads child exit incomplete')
    heads=pathlib.Path(heads);commit=heads/'commit_probe.pkl';boundary=heads/'boundary_probe.pkl'
    if not commit.is_file() or not boundary.is_file() or commit.is_symlink() or boundary.is_symlink():raise ValueError('missing original fitted heads')
    payload={'round':config['outer_step'],'encoder':config['dump_args']['model'],'commit':str(commit.resolve()),'boundary':str(boundary.resolve())}
    receipt={'scope':'ORIGINAL_G1_STAGED_CALIBRATION_COMPLETE_UNREVIEWED_NO_TRAINING','source_train_sha256':TRAIN_SHA,'owner_result_sha256':sha(pathlib.Path(directory)/'result.json'),'head_exit':head_exit,'commit_sha256':sha(commit),'boundary_sha256':sha(boundary),'manifest_payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
    # Persist evidence first. Publishing the live manifest is the final mutation.
    directory=pathlib.Path(directory);target=pathlib.Path(manifest)
    if target.is_symlink():raise ValueError('live manifest symlink rejected')
    old=target.read_bytes() if target.exists() else None
    receipt['publication_state']='PREPARED_NOT_ACCEPTED_VERIFY_LIVE_MANIFEST_SHA'
    encoded=json.dumps(payload,sort_keys=True).encode()
    receipt['manifest_file_sha256']=hashlib.sha256(encoded).hexdigest()
    evidence=directory/'calibration-publication-receipt.json'
    with open(evidence,'x') as f:
        os.chmod(evidence,0o600);json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    tmp=pathlib.Path(str(target)+'.staged.tmp');replaced=False
    try:
        fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(encoded);f.flush();os.fsync(f.fileno())
        os.replace(tmp,target);replaced=True
        fd=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    except BaseException:
        # Preserve last complete evaluator even if directory fsync failed after rename.
        if replaced:
            if old is None:target.unlink()
            else:
                rollback=pathlib.Path(str(target)+'.rollback.tmp')
                fd=os.open(rollback,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'wb') as f:f.write(old);f.flush();os.fsync(f.fileno())
                os.replace(rollback,target)
        if tmp.exists():tmp.unlink()
        raise
    return receipt
