"""One-shot CP5 continuation; only search stop and admitted eight-GPU endpoint change."""
import argparse,hashlib,json,os,shlex,socket,subprocess,time,uuid
from pathlib import Path
from omegaconf import OmegaConf
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--source',required=True);p.add_argument('--gate',required=True);a=p.parse_args()
b=Path(a.base).resolve();source=Path(a.source).resolve();gate=Path(a.gate).resolve()
assert os.getuid()==1005 and socket.gethostname().split('.')[0]=='xielab1'
assert source.parent==b and source.name.startswith('source-eight-') and not Path('/proc/3920441').exists()
stop=json.loads((b/'receipts/sparse-schedule/eight-gpu-boundary5-stop.json').read_text());assert stop['owned_release']
mf=b/'checkpoints/training_state/global_step_5/manifest.json';raw=mf.read_bytes();m=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()==stop['checkpoint_sha256']
g=json.loads(gate.read_text());assert g['accepted'] and g['scope']=='eight_gpu_full_scan_and_http_oracle_equality'
cp=Path('/proc',str(g['candidate_pid']));assert cp.stat().st_uid==1005
assert int((cp/'stat').read_text().rsplit(') ',1)[1].split()[19])==g['candidate_starttick']
apps={int(v) for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()}
assert apps<={2239529,g['candidate_pid']}
free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).split()]
assert len(free)==8 and min(free)>=20000
commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip();run=uuid.uuid4().hex
old=json.loads((b/'receipts/timed-guard-config-3bdfeb78c2a7480980f190a5c1c96bdf.json').read_text())
parent=Path(old['config']);assert hashlib.sha256(parent.read_bytes()).hexdigest()==m['identity']['config_sha256']
config=OmegaConf.load(parent);config.actor_rollout_ref.rollout.tree_action_tag_stop=True
config.actor_rollout_ref.rollout.tree_action_stop_mode='search_only';config.retriever.url='http://10.120.17.101:18006/retrieve'
folder=b/'receipts'/('eight-search-'+run);folder.mkdir()
configpath=folder/'resolved.yaml';OmegaConf.save(config,configpath)
binding=dict(protocol='recursive_a_sparse_schedule_continuation_v1',handoff_contract='search_only_and_no_repeated_candidate_proof_v1',
 parent_completed_step=5,parent_optimizer_calls=next(iter(m['identity']['guard_continue_accounting']['total_optimizer_calls_per_rank'].values())),
 parent_manifest=str(mf),parent_manifest_sha256=hashlib.sha256(raw).hexdigest(),parent_identity=m['identity'],
 parent_commit=m['identity']['code_commit'],parent_config=str(parent),parent_config_sha256=m['identity']['config_sha256'],
 eight_gpu_retrieval_gate=str(gate),eight_gpu_retrieval_gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),child_retrieval_url=config.retriever.url)
bp=folder/'binding.json';bp.write_text(json.dumps(binding,indent=2))
r=dict(old,run_id=run,source=str(source),source_commit=commit,config=str(configpath),config_sha256=hashlib.sha256(configpath.read_bytes()).hexdigest(),
 parent_completed_step=5,parent_actual_adam=binding['parent_optimizer_calls'],output=str(b/('output-search-only-resume5-'+run)))
rp=folder/'launch-config.json';rp.write_text(json.dumps(r,indent=2));log=b/'logs'/('search-only-eight-'+run+'.log')
python='/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python'
env=dict(RECURSIVE_BRAN_BASE=str(b),RECURSIVE_A_CHILD_CONFIG_RECEIPT=str(rp),RECURSIVE_A_CONTINUATION_BINDING=str(bp),
 RECURSIVE_A_CONTINUATION_SHA256=hashlib.sha256(bp.read_bytes()).hexdigest(),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONPATH=str(source),
 PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',RAY_DEDUP_LOGS='0')
command='cd '+shlex.quote(str(source))+'\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
command+='exec '+shlex.join([python,'-u','-m','search_r1.baselines.preflight','--',python,'-u',str(source/'release/recursive_bran_a/continue_search_only.py')])+' >'+shlex.quote(str(log))+' 2>&1\n'
launch=folder/'launch.sh';launch.write_text(command);launch.chmod(0o700)
session='rba-eight-search-'+run[:8];subprocess.run(['tmux','new-session','-d','-s',session,'bash '+shlex.quote(str(launch))],check=True)
subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
proof=dict(time=time.time(),run_id=run,source_commit=commit,config=str(rp),source=str(source),log=str(log),session=session,
 output=r['output'],parent_completed_outer=5,parent_actual_adam=r['parent_actual_adam'],gate=str(gate),status='STARTED_NOT_ACCEPTED')
(folder/'launch.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
