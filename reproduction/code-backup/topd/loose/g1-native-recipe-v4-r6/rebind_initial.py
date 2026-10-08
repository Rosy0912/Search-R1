"""Source-only R4 rebind of actual native3102 assembly. No model rehash/load/GPU."""
import argparse,copy,hashlib,json,os,pathlib
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
OLD_CANONICAL='ff091039ace0beb5ee56c93783c4b783a7359b85d73c07d04ec2d196e2d4266d'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def canonical(c):return hashlib.sha256(json.dumps({k:v for k,v in c.items() if k!='independent_gate'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def derive(old,pipeline,manifest_sha,gate):
 if canonical(old)!=OLD_CANONICAL:raise ValueError('Actual3102 whole static recipe drift')
 c=copy.deepcopy(old);c['pipeline_manifest_sha256']=manifest_sha
 c['producer']=str(ROOT/'tools/g1-staged-calibration-producer-v5');c['producer_manifest_sha256']='7f8940eb20a851adb3db665a9eabc59013979c0ee5588c1e79b67d806025a6ae'
 for b in c['source_bindings']:
  if b['kind']=='producer':b.update(root=c['producer'],manifest={'path':str(pathlib.Path(c['producer'])/'files.json'),'sha256':c['producer_manifest_sha256']})
  if b['kind']=='pipeline':b.update(root=str(pipeline),manifest={'path':str(pathlib.Path(pipeline)/'files.json'),'sha256':manifest_sha})
 c['independent_gate']={'path':str(gate),'sha256':None}
 c['retriever_health_url']=c['retriever_url'].rsplit('/',1)[0]+'/docs'
 c['evidence_root']=str(ROOT/'logs/g1-initial3102-r4r6-allocation-evidence')
 c['retriever_command'][3]=str(pathlib.Path(pipeline)/'retrieval_server.py')
 # All failed3120 outputs survive in their original paths. Original refit cache
 # must not skip generation or consume any earlier phase-A/head artifacts.
 work=ROOT/'logs/g1-initial3102-r4r6-calibration-work'
 c['refit_args']['work_dir']=str(work)
 c['refit_args']['manifest_root']=str(ROOT/'logs/g1-initial3102-r4r6-probe-manifests')
 index=c['probe_command'].index('--manifest')+1
 c['probe_command'][index]=str(pathlib.Path(c['refit_args']['manifest_root'])/('live_'+c['experiment'])/'manifest.json')
 changed={k:{'before':old[k],'after':c[k]} for k in c if old[k]!=c[k]}
 if set(changed)!={'producer','producer_manifest_sha256','pipeline_manifest_sha256','source_bindings','independent_gate','evidence_root','retriever_health_url','retriever_command','refit_args','probe_command'}:raise ValueError('Unexpected source-rebind delta')
 return c,changed

def atomic(p,value):
 with open(p,'x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--original-config',required=True);ap.add_argument('--assembly-receipt',required=True);ap.add_argument('--output',required=True);ap.add_argument('--pipeline-manifest-sha256',required=True);ap.add_argument('--producer-manifest-sha256',required=True);args=ap.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source rebind required')
 oldp=pathlib.Path(args.original_config);assembly=pathlib.Path(args.assembly_receipt);output=pathlib.Path(args.output);pipeline=ROOT/'tools/g1-production-pipeline-v3-r4-r6'
 for p in (oldp,assembly,output):
  if ROOT/'logs' not in p.resolve().parents or p.is_symlink():raise ValueError('ownlogs nonalias required')
 if output.exists():raise ValueError('Unique source rebind output required')
 a=json.loads(assembly.read_text())
 if a['status']!='ACTUAL_NATIVE_CPU_STATIC_RECIPE_NOT_GPU_RESOURCE_PROOF' or a['reviewed_static_recipe_sha256']!=OLD_CANONICAL or a['correct']!=17 or a['calibration_questions']!=300:raise ValueError('Actual3102 assembly receipt mismatch')
 producer=ROOT/'tools/g1-staged-calibration-producer-v5'
 if args.producer_manifest_sha256!='7f8940eb20a851adb3db665a9eabc59013979c0ee5588c1e79b67d806025a6ae' or sha(producer/'files.json')!=args.producer_manifest_sha256:raise ValueError('Stage cleanup producer frozen manifest drift')
 pfiles=json.loads((producer/'files.json').read_text())['files'];pactual=set()
 for p in producer.rglob('*'):
  if p.is_symlink():raise ValueError('Producer source alias')
  if p.is_file():pactual.add(str(p.relative_to(producer)))
 if pactual!=set(pfiles)|{'files.json'}:raise ValueError('Producer exact source set drift')
 for n,r in pfiles.items():
  p=producer/n
  if p.stat().st_size!=r['size'] or sha(p)!=r['sha256']:raise ValueError('Producer source bytes drift')
 manifest=pipeline/'files.json'
 if sha(manifest)!=args.pipeline_manifest_sha256:raise ValueError('R4 actual source manifest drift')
 records=json.loads(manifest.read_text())['files'];actual=set()
 for p in pipeline.rglob('*'):
  if p.is_symlink():raise ValueError('source alias')
  if p.is_file():actual.add(str(p.relative_to(pipeline)))
 if actual!=set(records)|{'files.json'}:raise ValueError('R4 exact source set drift')
 for n,r in records.items():
  p=pipeline/n
  if p.stat().st_size!=r['size'] or sha(p)!=r['sha256']:raise ValueError('R4 actual source bytes drift')
 if (ROOT/'logs/g1-initial3102-r4r6-calibration-work').exists() or (ROOT/'logs/g1-initial3102-r4r6-probe-manifests').exists() or (ROOT/'logs/g1-initial3102-r4r6-allocation-evidence').exists():raise ValueError('Fresh initial work/manifest/evidence roots required; failedA reuse forbidden')
 old=json.loads(oldp.read_text());c,changed=derive(old,pipeline,args.pipeline_manifest_sha256,output.with_name('g1-initial3102-r4r6-source-allocation-gate.json'))
 atomic(output,c)
 atomic(output.with_suffix('.source-rebind.json'),{'scope':'SOURCE_ONLY_REBIND_ACTUAL3102_REFS_NOT_NEW_NATIVE64_NOT_FRESH_MODEL_HASH_NOT_GPU_PROOF','actual_original_config':{'path':str(oldp),'sha256':sha(oldp),'canonical':canonical(old)},'actual3102_assembly':{'path':str(assembly),'sha256':sha(assembly)},'pipeline_manifest':{'path':str(manifest),'sha256':sha(manifest)},'producer_manifest':{'path':str(producer/'files.json'),'sha256':sha(producer/'files.json')},'exact_changed_fields':changed,'new_config':{'path':str(output),'sha256':sha(output),'canonical':canonical(c)},'independent_gate_pending':True})
 print(json.dumps({'scope':'SOURCE_ONLY_REBIND','config':str(output),'canonical':canonical(c),'independent_gate_pending':True}))
if __name__=='__main__':main()
