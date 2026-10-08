"""Concrete profile2/formal100 config from the same completed initial calibration.
CPU-only; no process launch, model load, allocation or approval generation.
"""
import argparse,copy,json,pathlib
from checkpoint_inventory import sha
from pipeline import atomic
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def derive(initial,mode,destination,pipeline_root,frozen_reference):
 if mode not in ('profile2','train100'):raise ValueError('profile2 or train100 required')
 destination=pathlib.Path(destination);pipeline_root=pathlib.Path(pipeline_root)
 for p in (destination,pipeline_root):
  if ROOT not in p.resolve().parents:raise ValueError('own local paths required')
 producer=ROOT/'tools/g1-staged-calibration-producer-v7'
 expected='671ec8d06f7b081119f7eda47deaf0db72ce33a721cf617799da80e88d3ef64b'
 producer_bindings=[b for b in initial['source_bindings'] if b['kind']=='producer']
 if initial.get('producer')!=str(producer) or initial.get('producer_manifest_sha256')!=expected or producer_bindings!=[{'kind':'producer','root':str(producer),'manifest':{'path':str(producer/'files.json'),'sha256':expected}}]:raise ValueError('Actual initial must bind final producerV7 source; failed old A reuse forbidden')
 config=copy.deepcopy(initial)
 from compiler_headers import binding
 config['compiler_headers']=binding()
 config['experiment']='search_g1_32b_'+mode
 config['evidence_root']=str(destination/'evidence');config['checkpoint_root']=str(destination/'checkpoints')
 step=2 if mode=='profile2' else 100
 config['final_checkpoint']=str(destination/'checkpoints/actor'/('global_step_'+str(step)))
 config['actual_outer_receipt_path']=str(destination/('actual-outer%d.json'%step))
 # SAME official base, fresh optimizer, original data/RNG path; not checkpoint2 resume.
 config['initial_calibration_reuse']=copy.deepcopy(frozen_reference)
 config['retriever_command'][3]=str(pipeline_root/'retrieval_server.py')
 config['refit_args'].update(work_dir=str(destination/'calibration-work'),manifest_root=str(destination/'probe-manifests'))
 manifest=pipeline_root/'files.json';config['pipeline_manifest_sha256']=sha(manifest)
 for b in config['source_bindings']:
  if b['kind']=='pipeline':b.update(root=str(pipeline_root),manifest={'path':str(manifest),'sha256':sha(manifest)})
 changes={'trainer.experiment_name':config['experiment'],'trainer.default_local_dir':config['checkpoint_root'],'trainer.total_training_steps':str(step+1),'trainer.save_freq':'2' if mode=='profile2' else '25'}
 command=[]
 for token in config['training_command']:
  key=token.split('=',1)[0].lstrip('+')
  command.append(key+'='+changes.pop(key) if key in changes else token)
 if changes:raise ValueError('missing original training override '+str(changes))
 config['training_command']=command
 probe=list(config['probe_command']);i=probe.index('--manifest')+1;probe[i]=str(destination/'probe-manifests'/('live_'+config['experiment'])/'manifest.json');config['probe_command']=probe
 config['independent_gate']={'path':str(destination/('profile2-source-allocation-gate.json' if mode=='profile2' else 'formal100-actual-update-resource-gate.json')),'sha256':None}
 config['run_scope']='PROFILE2_TWO_REAL_OUTERS_NOT_FORMAL100_NOT_TOPD' if mode=='profile2' else 'FORMAL100_FROM_OFFICIAL_BASE_WITH_SAME_BASE_ROUND0_HEADS_NOT_PROFILE_CHECKPOINT_RESUME'
 return config

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--initial-config',required=True);ap.add_argument('--mode',choices=['profile2','train100'],required=True);ap.add_argument('--directory',required=True);ap.add_argument('--initial-frozen',required=True);ap.add_argument('--initial-review',required=True);args=ap.parse_args()
 directory=pathlib.Path(args.directory).resolve()
 if ROOT/'logs' not in directory.parents or directory.exists():raise ValueError('fresh ownlogs required')
 frozen=pathlib.Path(args.initial_frozen);review=pathlib.Path(args.initial_review)
 for p in (frozen,review):
  if ROOT/'logs' not in p.resolve().parents or p.is_symlink():raise ValueError('actual native initial evidence required')
 r=json.loads(review.read_text())
 if r.get('status')!='PASS' or r.get('reviewed_frozen_calibration_sha256')!=sha(frozen):raise ValueError('actual initial independent review pending')
 reference={'path':str(frozen),'sha256':sha(frozen),'independent_review':{'path':str(review),'sha256':sha(review)}}
 config=derive(json.loads(pathlib.Path(args.initial_config).read_text()),args.mode,directory,pathlib.Path(__file__).resolve().parent,reference)
 directory.mkdir(mode=0o700);atomic(directory/'static-config.json',config)
 print(json.dumps({'scope':config['run_scope'],'config':str(directory/'static-config.json'),'independent_gate_pending':True}))
if __name__=='__main__':main()
