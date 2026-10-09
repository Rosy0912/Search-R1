"""HKUST3 direct campus transfer of a downloaded/forward-accepted archive."""
import fcntl,json,os,pathlib,sys,time
import migrate_topd_to_hpc3_stream as relay
root=pathlib.Path(__file__).resolve().parent
name=sys.argv[1];plan=json.loads((root/'missing-plan.json').read_text());record=next(x for x in plan['records'] if x['group']==name)
lock=(root/(name+'-hpc3.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
relay.LOCAL=root/'progress';relay.LOCAL.mkdir(exist_ok=True)
ssh=['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn','-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts','-oConnectTimeout=8','jxu957@10.120.81.30'];relay.ssh_args=lambda command:ssh+[command]
record['archive']='/backup/shaoshitong/topd-hf-migration-20261009/hkust2-forward/'+name+'.tar.zst'
r=json.loads(pathlib.Path(record['archive']).with_name(name+'-forward-verified.json').read_text());assert r['sha256']==record['archive_sha256'] and r['bytes']==record['archive_bytes']
(root/(name+'-controller.json')).write_text(json.dumps({'pid':os.getpid(),'epoch':time.time(),'group':name,'route':'HKUST3->HPC3','mac_used':False}))
print(json.dumps(relay.transfer(record,source_reader=['sh'])),flush=True)
