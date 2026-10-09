"""Run the existing archive/file verifier on HKUST3, with direct campus SSH."""
import fcntl, json, pathlib, shlex, time, os
import migrate_topd_to_hpc3_stream as relay
root=pathlib.Path(__file__).resolve().parent
relay.LOCAL=root/'progress';relay.LOCAL.mkdir(exist_ok=True)
lock=(root/'wiki-hpc3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ssh=['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn','-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts','-oConnectTimeout=8','jxu957@10.120.81.30']
relay.ssh_args=lambda command:ssh+[command]
record=json.loads((root/'wiki-plan.json').read_text())
record['archive']='/backup/shaoshitong/topd-hf-migration-20261009/hkust2-forward/wiki_corpus.tar.zst'
receipt=json.loads(pathlib.Path(record['archive']).with_name('wiki_corpus-forward-verified.json').read_text())
assert receipt['sha256']==record['archive_sha256']
(root/'controller.json').write_text(json.dumps({'pid':os.getpid(),'epoch':time.time(),'route':'HKUST3->HPC3','mac_used':False,'group':'wiki_corpus'}))
print(json.dumps(relay.transfer(record,source_reader=['sh'])),flush=True)
