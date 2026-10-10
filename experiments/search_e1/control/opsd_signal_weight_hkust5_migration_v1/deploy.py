"""Deploy only small code/provenance from Mac; payload was transferred node to node."""
import subprocess,json,base64,hashlib,shlex,sys
from pathlib import Path
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ProxyCommand=none','-o','ProxyJump=none','-o','ControlMaster=no','-o','ControlPath=none']
R=Path(__file__).resolve().parent
SOURCE=r'''import pathlib,json,hashlib,base64
S=pathlib.Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010');H=S/'group-H_POS/runs'
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
files={str(S/('group-'+g)/'runs/training-em-25-complete.json'):base64.b64encode((S/('group-'+g)/'runs/training-em-25-complete.json').read_bytes()).decode() for g in ['G_POS','G_NEG']}
cp=json.loads((H/'outers/outer-00001/training/outer-complete.json').read_text())['checkpoint'];cursor=json.loads((H/'outers/outer-00001/outer-manifest.json').read_text())['train_cursor']['after'];stop=sorted((S/'preserved').glob('migration-hkust5-*/stop-result.json'))[-1]
print(json.dumps(dict(files=files,resume_checkpoint=cp,resume_cursor=cursor,previous_hpos_summaries=[ref(H/'outers'/('outer-%05d'%i)/'summary.json') for i in range(2)],completed_groups=[dict(variant=g,ledger=ref(S/('group-'+g)/'runs/training-em-25-complete.json')) for g in ['G_POS','G_NEG']],source_stop=json.loads(stop.read_text()))))
'''
TARGET=r'''import sys,json,pathlib,base64,hashlib,shutil,time,os
P=json.load(sys.stdin);B=pathlib.Path('/home/shaoshitong/search-e1-migration-20261011');S=pathlib.Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010');host=lambda p:B/str(p).lstrip('/')
assert os.uname().nodename=='xie-lab-4' and P['transfer']['accepted']
root=S/'source/hkust5-migration-v1/opsd_signal_weight_hkust5_migration_v1';actual=host(root);actual.mkdir(parents=True,exist_ok=False)
for name,data in P['helper'].items():(actual/name).write_bytes(base64.b64decode(data))
assert hashlib.sha256((actual/'source-manifest.json').read_bytes()).hexdigest()==P['manifest_sha']
for name,digest in json.loads((actual/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((actual/name).read_bytes()).hexdigest()==digest
for path,data in P['source']['files'].items():
 p=host(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
O=host('/weka_3/shaoshitong/search-e1-config-study-20261004');(O/'env').symlink_to('/home/shaoshitong/search-e1-config-study-20261004/env',target_is_directory=True);(O/'experiment.lock').touch()
pending=host(S/'group-H_POS/runs/outers/outer-00002');assert not (pending/'sampling-complete.json').exists() and not list(pending.glob('sampling-attempt-*'))
preserve=host(S/'preserved/migration-unused-source-outer2');shutil.move(str(pending),str(preserve));pending.mkdir()
for name in ['questions.jsonl','train-selection.json']:shutil.copyfile(preserve/name,pending/name)
def write(path,value):
 raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();host(path).write_bytes(raw);return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
pub=write(S/'receipts/hkust5-migration-publication.json',dict(accepted=True,branch='search-e1',source_commit=P['commit'],remote_readback_sha=P['commit'],source_manifest_sha256=P['manifest_sha'],deployed_helper=str(root),all_deployed_files_read_back=True,checked_unix=time.time()))
binding={k:v for k,v in P['source'].items() if k!='files'};binding.update(schema='opsd-signal-same-state-host-migration-v1',accepted=True,human_authorization='可以的，迁移过去，去HKUST5跑。',publication=pub,helper_manifest=dict(path=str(root/'source-manifest.json'),sha256=P['manifest_sha']),target_node='HKUST5',target_output_realpath=str(host(S)),historical_namespace='/weka_3',environment_parity=P['environment'],transfer=P['transfer'],observed_unix=time.time(),training_first_tests_deferred=True,big_outputs_remote_only=True)
ref=write(S/'receipts/hkust5-migration-binding.json',binding);print(json.dumps(dict(accepted=True,binding=ref,publication=pub,target_realpath=str(host(S)),unused_source_outer2_preserved=str(preserve))))
'''
def main(commit,output):
 source=subprocess.run(SSH+['hkust4','python3 -'],input=SOURCE.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
 manifest=R/'source-manifest.json';names=list(json.loads(manifest.read_text())['files'])+['source-manifest.json'];files={n:base64.b64encode((R/n).read_bytes()).decode() for n in names};out=Path(output)
 env=json.loads((out.parent/'hkust5-environment-parity.json').read_text());assert env['all_package_python_and_binary_bytes_equal']
 payload=dict(source=json.loads(source.stdout),helper=files,commit=commit,manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest(),environment=env,transfer=json.loads((out.parent/'hkust5-direct-transfer-terminal.json').read_text()))
 r=subprocess.run(SSH+['hkust5','python3 -c '+shlex.quote(TARGET)],input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 value=dict(actual_wait=r.returncode,stderr=r.stderr.decode(errors='replace'),result=json.loads(r.stdout) if r.returncode==0 else r.stdout.decode(errors='replace'),commit=commit,local_large_files_created=0)
 out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value));assert r.returncode==0
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
