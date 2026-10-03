"""Fit both actual G1 heads and publish one immutable encoder/readout version."""
import argparse,hashlib,json,os,subprocess,time,uuid
from pathlib import Path
import joblib,numpy as np
from search_g1.resource import evaluator_gpu_lock
from reproduction.a800.prepare_run import digest

ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,required=True);ap.add_argument('--model',required=True);args=ap.parse_args()
base=Path(os.environ.get('SEARCH_G1_BASE','/data/search-g1'));run=json.loads((base/'logs/run-manifest.json').read_text())
assert args.step in (0,50,100,150,200,250)
if args.step:
 complete=json.loads((Path(args.model)/'complete.json').read_text());assert complete['run_id']==run['run_id'] and complete['step']==args.step
encoder=Path(args.model)
index=json.loads((encoder/'model.safetensors.index.json').read_text())
names=set(index['weight_map'].values())|{'model.safetensors.index.json','config.json','tokenizer.json','tokenizer_config.json'}
encoder_hashes={name:digest(encoder/name) for name in sorted(names)}
expected=complete['files_sha256'] if args.step else run['initial_model_sha256']
assert all(encoder_hashes[name]==value for name,value in expected.items() if name in encoder_hashes), 'Encoder changed before calibration'
round_dir=base/'artifacts'/('round_'+str(args.step)+'-attempt-'+uuid.uuid4().hex);round_dir.mkdir(parents=True,exist_ok=False)
dump=round_dir/'dumps';heads=round_dir/'heads';env=dict(os.environ,RETRIEVER_URL='http://127.0.0.1:18010/retrieve',NO_PROXY='127.0.0.1,localhost')
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(k,None)
python=str(base/'env/bin/python');repo=base/'repo'
with evaluator_gpu_lock():
 with (round_dir/'calibration.log').open('w') as log:
  subprocess.run([python,str(repo/'ts_probe_dump.py'),'--model',args.model,'--data',str(base/'data/calibration-300.parquet'),'--tag','nq-hotpot-common','--cap','300','--G','5','--layers','21,24,27','--gpu_mem','0.25','--out',str(dump)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([python,str(repo/'ts_probe_train.py'),'--dumps',str(dump),'--layers','21,24,27','--commit-position','Xa','--commit-label','flip','--save',str(heads)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 report=json.loads((heads/'calibration-report.json').read_text());proof=json.loads((dump/'calibration-label-proof.json').read_text())
 assert proof['question_count']==300 and report['unique_question_count']==300
 assert np.isfinite(report['commit_cv_auc']) and np.isfinite(report['boundary_cv_auc'])
 assert min(report['commit_class_counts'])>0 and min(report['boundary_class_counts'])>0
 for name in ['commit_probe.pkl','boundary_probe.pkl']:
  obj=joblib.load(heads/name);assert list(obj['clf'].classes_)==[0,1] and np.isfinite(obj['auc'])
 for L in (21,24,27):
  for file in dump.glob('*.npz'):
   with np.load(file,allow_pickle=False) as z:
    for key in ('Xa_'+str(L),'Xp_question_'+str(L)):assert np.isfinite(z[key]).all()
 version=run['run_id']+':'+str(args.step)+':'+uuid.uuid4().hex
 payload={'run_id':run['run_id'],'round':args.step,'version':version,'encoder':str(Path(args.model).resolve()),'commit':str(heads/'commit_probe.pkl'),'boundary':str(heads/'boundary_probe.pkl'),'encoder_sha256':encoder_hashes,'head_sha256':{n:hashlib.sha256((heads/n).read_bytes()).hexdigest() for n in ['commit_probe.pkl','boundary_probe.pkl']},'calibration_report':report,'label_proof':proof,'published_unix':time.time()}
 live=base/'artifacts/live';live.mkdir(exist_ok=True);tmp=live/'manifest.json.tmp';tmp.write_text(json.dumps(payload,indent=2));os.replace(tmp,live/'manifest.json')
 print('SEARCH_G1_BOTH_PROBES_FITTED='+version,flush=True)
