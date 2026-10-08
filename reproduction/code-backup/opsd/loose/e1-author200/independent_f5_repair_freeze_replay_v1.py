"""Local-only byte/lineage closure check of the new immutable CPU repair tools."""
from pathlib import Path
import hashlib
import json
import stat

HERE = Path(__file__).resolve().parent
ROOT = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006'
RUN = ROOT + '/runs/formal-3261'
PIN = 'f5c72c9426413a18e465030efb714c9015fa84274baa3fc4e401383adc96b3fd'
OLDPIN = '8441e10fa4565bf53f7baf24a57a60ddbe9844dc979a2c9a0c97250219df7019'
FROZEN = HERE / 'final-acceptance-f5c72c942641'
OLD = HERE / 'final-acceptance-8441e10fa456'

def rawpin(path, pin):
    assert path.is_file() and not path.is_symlink() and path.resolve() == path
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin, path
    return raw

def main():
    manifest = json.loads(rawpin(FROZEN/'manifest.json', PIN))
    old = json.loads(rawpin(OLD/'manifest.json', OLDPIN))
    assert type(manifest['version']) is int and manifest['version'] == 1
    assert manifest['source_commit'] == '188f4e5e7b63cf479290d0888a4a075499886750'
    assert manifest['kind'] == old['kind'] and manifest['scope'] == old['scope']
    assert manifest['contract'] == old['contract']
    expected_tools = dict(old['tools'])
    expected_tools['inspect_author200_checkpoint_cpu.py'] = 'f7d8482e54c727c878a3114d4d748889e4bcd38cc7dac573000f6ba2cda1864a'
    expected_tools['checkpoint_repair_cpu.py'] = 'a740a5106b7b8c338da294d14cd6437a711ac79e74b50b28770ed9814b5a60e2'
    assert manifest['tools'] == expected_tools
    assert len(manifest['review_records']) == 12
    for name, pin in old['review_records'].items():
        assert manifest['review_records'][name] == pin
    memberpins = manifest['tools'] | manifest['review_records'] | {'manifest.json':PIN}
    assert len(memberpins) == 21 and {p.name for p in FROZEN.iterdir()} == set(memberpins)
    inode_keys=set()
    for name, pin in memberpins.items():
        p=FROZEN/name;rawpin(p,pin);st=p.stat()
        assert stat.S_IMODE(st.st_mode) == 0o444 and st.st_nlink == 1
        assert (st.st_dev,st.st_ino) not in inode_keys
        inode_keys.add((st.st_dev,st.st_ino))
    assert stat.S_IMODE(FROZEN.stat().st_mode) == 0o555
    expected_deps=dict(old['dependencies'])
    expected_deps.update({ROOT+'/tools/final-acceptance-8441e10fa456/'+n:p for n,p in old['tools'].items()})
    expected_deps[ROOT+'/tools/final-acceptance-8441e10fa456/manifest.json']=OLDPIN
    original = HERE/'actual-snapshots/3261-20261005T224639356135Z/original-files/evidence'
    markerpin='feac6e0c56a35e7725307b0d65c6f0437a0ae18bba13bea07e774721b1a2e88f'
    inventorypin='32f2c38c121b0c59ab26036c358b4fd749218f8e3980d6700f54c78fabbe6c4e'
    marker=json.loads(rawpin(original/'checkpoint-cpu-25-launch.json',markerpin))
    inventory=json.loads(rawpin(original/'checkpoint-25-cpu-inspection-once-v1/observation/observed-checkpoint-inventory.json',inventorypin))
    assert type(marker['checkpoint_step']) is int and marker['checkpoint_step']==25
    assert marker['tools_manifest_sha256']==OLDPIN
    assert inventory['kind']=='OBSERVED_AUTHOR200_CHECKPOINT_INVENTORY_NOT_TRAINING_MANIFEST'
    assert inventory['checkpoint_step']==25 and inventory['raw_statuses']=={'driver':'DRIVER_STATE_SERIALIZED_OK','ranks':['FULL_STATE_SERIALIZED_OK']*8}
    expected_deps[RUN+'/evidence/checkpoint-cpu-25-launch.json']=markerpin
    expected_deps[RUN+'/evidence/checkpoint-25-cpu-inspection-once-v1/observation/observed-checkpoint-inventory.json']=inventorypin
    status=json.loads((HERE/'checkpoint25-native-status-20261005T222408Z-original.json').read_bytes())
    streams=json.loads((HERE/'checkpoint25-cpu-failure-streams-original.json').read_bytes())
    recovered=HERE/'repair25-lineage-exact-recovered-v1'
    for name,item in status['files'].items():
        blob=rawpin(recovered/name,item['sha256'])
        assert len(blob)==item['bytes'] and json.loads(blob)==item['content']
        assert blob==(json.dumps(item['content'],sort_keys=True,separators=(',',':'))+'\n').encode()
        assert type(item['content']['returncode']) is int and item['content']['returncode']==1
        expected_deps[RUN+'/evidence/checkpoint-25-cpu-inspection-once-v1/'+name]=item['sha256']
    item=streams['files']['native-stderr-original.txt']
    blob=rawpin(recovered/'native-stderr-original.txt',item['sha256'])
    assert len(blob)==item['bytes'] and blob==item['content'].encode()
    assert blob.endswith(b'ValueError: Sampler position/config differs: seed\n')
    expected_deps[RUN+'/evidence/checkpoint-25-cpu-inspection-once-v1/native-stderr-original.txt']=item['sha256']
    assert len(expected_deps)==268 and manifest['dependencies']==expected_deps
    receipt=json.loads((HERE/'final-acceptance-freeze-f5c72c942641.json').read_bytes())
    assert receipt['manifest_sha256']==PIN and receipt['previous_manifest_sha256']==OLDPIN
    assert receipt['dependency_count']==268 and receipt['unchanged_original_dependency_count']==254
    assert receipt['remote_deployment_performed'] is False and receipt['actual_checkpoint25_content_acceptance'] is False
    assert receipt['new_dependency_union']=={k:v for k,v in expected_deps.items() if k not in old['dependencies']}
    # This is root's newly copied actual remote read, not inherited254 proof.
    remote=HERE/'final-acceptance-deployment-f5c72c942641-originals/before-dependency-check-original.json'
    read=json.loads(rawpin(remote,'c4d76a1a7f2e3ff694821d25f016338ba9ada404c9b58fad4080de3703552edb'))
    assert read['source_commit']==manifest['source_commit'] and read['source_clean'] is True
    assert len(read['dependencies'])==268 and {v['path']:v['sha256'] for v in read['dependencies']}==expected_deps
    print(json.dumps({'status':'PASS_FROZEN_REPAIR_TOOLS_AND_COPIED_FRESH268_NOT_CPU_CONTENT','manifest_sha256':PIN,'tools':8,'reviews':12,'members':21,'dependencies':268,'exact_old254_plus_failed_lineage14':True,'failure_wire_exact_bytes_verified':True,'old8441_native_srun_controller_all1_preserved':True,'raw_serialized_statuses_preserved':True,'fresh_remote268_original_sha256':hashlib.sha256(remote.read_bytes()).hexdigest(),'actual25_repaired_content_pass':False,'limitations':['Local read-only replay, not reviewer remote execution','New freeze and fresh dependency record do not prove tool deployment or repair CPU completion','Only actual CPU native report plus waited native/srun/controller exits and independent actual audit can accept25']}))

if __name__=='__main__': main()
