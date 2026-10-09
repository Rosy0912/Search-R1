"""Hold the supplement writer reservation while its authorized HF route runs.

No asset bytes, SSH credentials, GPU allocation, or training are handled here.
Only a separately produced fresh destination verification receipt completes it.
"""
import fcntl, json, os, pathlib, time
BASE=pathlib.Path(__file__).resolve().parents[1]
E=BASE/'outputs/evidence'
lock=(E/'native-x-hpc-retriever-supplement.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state_path=E/'native-x-hpc-retriever-supplement-progress.json'
if state_path.exists():
    previous=json.loads(state_path.read_text())
    (E/('native-x-hpc-supplement-before-hf-guard-'+str(time.time_ns())+'.json')).write_text(json.dumps(previous,indent=2))
state={'pid':os.getpid(),'started_epoch':time.time(),'status':'waiting_hf_route_destination_verification','route':'H100 -> mgmt RAM -> private HF -> HKUST2 -> HKUST3 -> HPC3','verified_files':0,'verified_bytes':0,'training_started':False,'writer_reservation_only':True}
while True:
    receipt=E/'native-x-hpc-hf-supplement-target-fresh-verified.json'
    if receipt.exists():
        result=json.loads(receipt.read_text())
        assert result['status']=='SUPPLEMENT_ALL_ORIGINAL_FILES_FRESH_SHA_VERIFIED'
        assert result['verified_files']==39714 and result['verified_bytes']==7529241082
        assert result['archive_sha256']=='edc951c6a9cade2134ca0badc58b03852aae800b73d55d498d1a4c3a95331787'
        assert result['fresh_original_files_verified'] is True
        assert all(result['hop_archive_sha_verified'].get(h) is True for h in ['source','hf','hkust2','hkust3','hpc3'])
        state.update(status='asset_transfer_complete_all_sha_verified',verified_files=result['verified_files'],verified_bytes=result['verified_bytes'],receipt=str(receipt))
    state['updated_epoch']=time.time()
    tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,state_path)
    if state['status']=='asset_transfer_complete_all_sha_verified':break
    time.sleep(5)
