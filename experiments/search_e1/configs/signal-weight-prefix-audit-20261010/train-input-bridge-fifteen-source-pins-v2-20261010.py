
import hashlib,json,os,sys,time
from pathlib import Path
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
CFG_SHA='e7c0659bc6544d3440b7f67b1ba2f976c56c79d8371e23b4dbe614cbb894f661'
NATIVE_SHA='af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314'
COMPLETION_SHA='b36aff538310bbd92c947e790e32254ce3267e9f1f62531aa4729a74bc719163'
PUB_SHA='0dc1375d0d760fd57d98bbf09e05ca512c78edd094f6281bb44af91515dee0d7'
def require(x,m):
 if not x: raise ValueError(m)
def stat(p):
 q=p.stat();return dict(device=q.st_dev,inode=q.st_ino,bytes=q.st_size,mtime_ns=q.st_mtime_ns,ctime_ns=q.st_ctime_ns)
def fresh(p,sha=None):
 p=Path(p).resolve();before=stat(p);raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
 require(before==stat(p) and (sha is None or h==sha),'Changed metadata/SHA: '+str(p));return dict(path=str(p),sha256=h,**before),raw
def small(ref): return {k:ref[k] for k in ('path','sha256','bytes')}
def readpath(p,sha=None):
 ref,raw=fresh(p,sha);value=json.loads(raw);require(stat(Path(ref['path']))=={k:ref[k] for k in ('device','inode','bytes','mtime_ns','ctime_ns')},'Parse race');return value,ref
def readref(ref):
 value,p=readpath(ref['path'],ref['sha256']);require(ref.get('bytes',p['bytes'])==p['bytes'],'Reference bytes mismatch');return value,p
require(os.getuid()==1001 and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules,'CPU reader contract')
nh=S/'source/training-00fec71/opsd_signal_weight_v1';ch=S/'source/completion-7d9fea2/opsd_signal_weight_completion_v1'
nm,np=readpath(nh/'source-manifest.json',NATIVE_SHA);cm,cp=readpath(ch/'source-manifest.json',COMPLETION_SHA)
require(len(nm['files'])==46 and len(cm['files'])==6,'Original source inventories differ');sources={}
for name,entry in cm['files'].items():
 p,_=fresh(ch/name,entry['sha256']);require(p['bytes']==entry['bytes'],'Completion source bytes changed');sources[name]=small(p)
pub,pp=readpath(S/'receipts/completion-publication-7d9fea2.json',PUB_SHA);require(pub.get('accepted') is True,'Publication not accepted')
cfg,cfgref=readpath(S/'group-G_POS/receipts/config.json',CFG_SHA)
require(cfg['variant']=='G_POS' and cfg['study_id']=='signal-four-1fd4da344d9147e7be99c62e5d7c2d26' and cfg['run_id']=='59de0d17968b4955b97f312cb2387429','Original config/run differs')
require(cfg['helper_source_manifest']['sha256']==NATIVE_SHA and Path(cfg['paths']['helper_root']).resolve()==nh,'Original native source differs')
base=Path(cfg['paths']['output_root']).resolve();require(base==S/'group-G_POS/runs','Unexpected output root');rows=[];previous=None;previous_updates=0
for index in range(15):
 outer=base/'outers'/('outer-%05d'%index);manifest,mp=readpath(outer/'outer-manifest.json');complete,co=readpath(outer/'training/outer-complete.json')
 require(manifest['schema']=='opsd-online-outer-manifest-v1' and manifest['outer_index']==index and Path(manifest['outer_dir']).resolve()==outer,'Actual outer path/index differs')
 require(manifest['run_id']==cfg['run_id'] and manifest['config_sha256']==CFG_SHA and manifest['variant']=='G_POS' and manifest['runtime_identity']==cfg['runtime_manifest']['sha256'] and manifest['source_model_identity']==cfg['source_model_manifest']['sha256'],'Outer config/model/runtime differs')
 require(manifest['previous_complete_checkpoint']==previous and manifest['cumulative_optimizer_updates_before']==previous_updates,'Cumulative native metadata differs')
 require(complete['schema']=='opsd-online-outer-complete-v1' and complete['accepted'] is True and complete['run_id']==cfg['run_id'] and complete['outer_index']==index and complete['P']==manifest['P'],'Actual completed outer missing')
 end=complete['checkpoint'];endbody,endactual=readref(end)
 require(endbody['run_id']==cfg['run_id'] and endbody['config_sha256']==CFG_SHA and endbody['cursor']['outer_index']==index and endbody['optimizer_updates']==complete['optimizer_updates'],'Raw end metadata differs')
 sampling,sp=readpath(Path(manifest['outer_dir'])/'sampling-complete.json');body=sampling['train_em'];exitproof=sampling['exit_proof'];terminal,tp=readref(exitproof['actual_terminal'])
 require(sampling['schema']=='opsd-online-sampling-complete-v1' and sampling['accepted'] is True and sampling['run_id']==cfg['run_id'] and sampling['rollouts']==manifest['rollouts'] and sampling['pairs']==manifest['pairs'] and sampling['sampling_model_identity']==manifest['sampling_model_identity'],'Sampling differs from native outer')
 require(body['accepted'] is True and body['variant']=='G_POS' and body['outer_index']==index and body['questions']==512 and body['unique_slots']==2560 and body['overall']['denominator']==2560 and body['measurement_time']=='outer_start_sampling_before_any_this_outer_update' and body['post_training_em'] is False,'Actual preupdate2560 receipt differs')
 require(exitproof['accepted'] is True and len(exitproof['child_exit_evidence'])==8 and terminal['child_exit_evidence']==exitproof['child_exit_evidence'] and terminal['run_id']==cfg['run_id'] and terminal['config_sha256']==CFG_SHA and terminal['phase']=='sampling-running' and exitproof['gpu_release']['all_owned_release']['accepted'] is True,'Sampling terminal/release metadata differs')
 require(all(x['actual_wait_returncode']==0 and x['identity']['uid']==1001 and x['identity']['run_id']==cfg['run_id'] and type(x['identity']['starttick']) is int for x in exitproof['child_exit_evidence']),'Actual sampling wait not0')
 target=S/'analysis/train-input-bridge-7d9fea2'/('G_POS-outer-%05d.json'%index);existing=None
 if target.exists():
  value,proof=readpath(target);expected=dict(body,config=small(cfgref),run_id=cfg['run_id'],original_sampling_complete=small(sp),input_format_bridge_only=True)
  require(value==expected,'Existing bridge differs; do not overwrite '+str(target));existing=small(proof)
  if index==0:require(existing['sha256']=='fd1661e8e06d6d700ccd922862ead06f8259050252fc8ef799ef6fa16e3e9799' and existing['bytes']==326241,'Accepted outer0 bridge changed')
 require(index!=0 or existing is not None,'Accepted outer0 bridge missing')
 rows.append(dict(variant='G_POS',outer_index=index,original_outer_manifest=mp,training_completion=co,sampling_complete=sp,actual_sampling_terminal=tp,original_raw_end_checkpoint=end,fresh_raw_end_metadata=endactual,original_sampling_model_sha256=manifest['sampling_model_identity']['sha256'],actual_sampling_child_wait_returncodes=[x['actual_wait_returncode'] for x in exitproof['child_exit_evidence']],actual_sampling_gpu_release_accepted=True,actual_complete_optimizer_updates=complete['optimizer_updates'],P=complete['P'],K=complete['K'],train_receipt_scope=dict(denominator=2560,measurement_time=body['measurement_time'],not_recomputed=True),bridge_output_path=str(target),existing_bridge=existing))
 previous=end;previous_updates=complete['optimizer_updates']
for proof in sources.values():fresh(proof['path'],proof['sha256'])
fresh(pp['path'],PUB_SHA);fresh(np['path'],NATIVE_SHA);fresh(cfgref['path'],CFG_SHA)
require('torch' not in sys.modules,'Torch imported')
print(json.dumps(dict(schema='opsd-signal-train-bridge-source-pins-v1',accepted=True,study_id=cfg['study_id'],variant=cfg['variant'],run_id=cfg['run_id'],checked_at_unix=time.time(),config=cfgref,source_commit='7d9fea2333d85716cce1bd08c3d10ee0f3248e28',source_manifest=cp,publication=pp,source_inventory=sources,native_source_manifest=np,complete_outer_indices=list(range(15)),complete_outers=15,outer0_bridge_reusable=True,bridge_actions_needed=sum(x['existing_bridge'] is None for x in rows[1:]),bridge_executed=False,new_node_outputs_created=False,gpu_http_model_rng_signals_used=False,no_raw_payload_transferred=True,scope='Read-only completed native sampling/terminal/end metadata pins and existing bridge identity only; no TRAIN EM recomputation, token/numeric audit, heldout completion or full100 analysis.',outers=rows),sort_keys=True,allow_nan=False))
