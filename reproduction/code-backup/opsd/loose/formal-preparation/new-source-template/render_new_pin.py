"""Formal stage only after actual pilot/HF IDs and picked micro. Never rewrite earlier stages."""
import hashlib,json,re,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from verify_formal_prerequisites import loads,RUNTIME_NAMES,COMMIT as OLD
from render_staged import source_profile,fresh_output,guard,sha,replace_once

def render(profile_path,output):
 p=source_profile(profile_path);commit=p['source_commit']
 for key in ('pilot_job','reload_job'):assert re.fullmatch('[0-9]+',p[key]) and p[key] not in ('2960','2962')
 assert p['pilot_job']!=p['reload_job']
 micro=p['picked_local_micro'];assert type(micro) is int and micro in (2,4,8)
 pilot_path=Path(p['pilot_runtime_manifest']);hf_path=Path(p['hf_runtime_manifest'])
 pm=loads(pilot_path.read_text());hm=loads(hf_path.read_text())
 assert pm['source_commit']==hm['source_commit']==commit
 assert pm['production_math_sources']==hm['production_math_sources']==p['production_math_sources']
 assert pm['stage']=='pilot_only_no_future_ids_or_picked' and hm['stage']=='hf_only_actual_pilot_no_future_reload'
 assert hm['pilot_job']==p['pilot_job'] and hm['pilot_runtime_manifest_sha256']==sha(pilot_path)
 assert hm['pilot_runtime_directory_name']==pm['directory_name']
 for path,m in ((pilot_path,pm),(hf_path,hm)):
  for n,v in m['file_sources'].items():assert sha(path.parent/n)==v
 assert set(pm['runtime_sources'])==RUNTIME_NAMES
 out=fresh_output(output);formal='formal-'+commit[:12]+'-pilot-'+p['pilot_job']+'-hf-'+p['reload_job']
 def rep(s):return s.replace(OLD,commit).replace('opsd-acceptance-2960','opsd-new-acceptance-'+p['pilot_job']).replace('opsd-hf-reload-2962','opsd-new-hf-reload-'+p['reload_job']).replace('opsd-gpu-acceptance-5',pm['checkpoint_experiment']).replace('2960',p['pilot_job']).replace('2962',p['reload_job']).replace('tools/runtime/',f'tools/{pm["directory_name"]}/').replace('tools/formal/',f'tools/{formal}/')
 for n in ('formal_entry.sh','formal_job.sh','formal_supervisor.py','verify_formal_prerequisites.py','submit_formal.sh','PLAN.md','final_evaluation_overrides.json'):
  original=(HERE.parent/n).read_text()
  critical={'formal_entry.sh':['python "$root/tools/formal/verify_formal_prerequisites.py"','bash "$root/tools/formal/formal_job.sh"'],'formal_job.sh':['python "$root/tools/formal/verify_formal_prerequisites.py" --cleanup','python -u "$root/tools/formal/formal_supervisor.py"']}
  for target in critical.get(n,[]):assert original.count(target)==1,(n,target,original.count(target))
  if n=='submit_formal.sh':
   # A retained accepted receipt survives Slurm MinJobAge; purged IDs do not.
   dependency=' --dependency=afterok:2962 --kill-on-invalid-dep=yes '+chr(92)+'\n'
   original=replace_once(original,dependency,'')
  s=rep(original)
  if n=='formal_entry.sh':
   s=replace_once(s,': "${OPSD_ACCEPTANCE_RECEIPT:',': "${OPSD_STAGE_MANIFEST_SHA256:?Externally reviewed FINAL manifest SHA required}"\n: "${OPSD_ACCEPTANCE_RECEIPT:')
   s=guard(s)
  if n=='submit_formal.sh':
   s=replace_once(s,': "${OPSD_ACCEPTANCE_RECEIPT:',': "${OPSD_STAGE_MANIFEST_SHA256:?Externally reviewed FINAL manifest SHA required}"\n: "${OPSD_ACCEPTANCE_RECEIPT:')
   s=replace_once(s,'--export=ALL,OPSD_ACCEPTANCE_RECEIPT="$OPSD_ACCEPTANCE_RECEIPT"','--export=ALL,OPSD_ACCEPTANCE_RECEIPT="$OPSD_ACCEPTANCE_RECEIPT",OPSD_STAGE_MANIFEST_SHA256="$OPSD_STAGE_MANIFEST_SHA256"')
  if n=='formal_job.sh':s=replace_once(s,'OPSD_GLOBAL_MICROBATCH=16',f'OPSD_GLOBAL_MICROBATCH={micro*8}')
  if n=='formal_supervisor.py':
   s=replace_once(s,'capacity_micro >= 2',f'capacity_micro >= {micro}')
   s=replace_once(s,'local_micro=2',f'local_micro={micro}')
   s=replace_once(s,'ppo_micro_batch_size==16',f'ppo_micro_batch_size=={micro*8}')
  if n=='verify_formal_prerequisites.py':
   s=s.replace("('runtime_sources',RUNTIME_NAMES,'runtime')",f"('runtime_sources',RUNTIME_NAMES,'{pm['directory_name']}')").replace("('formal_sources',FORMAL_NAMES,'formal')",f"('formal_sources',FORMAL_NAMES,'{formal}')")
   s=s.replace("FORMAL_NAMES=set('formal_entry.sh formal_job.sh formal_supervisor.py verify_formal_prerequisites.py'.split())","FORMAL_NAMES=set('formal_entry.sh formal_job.sh formal_supervisor.py verify_formal_prerequisites.py micro_evidence.py collect_receipt.py controller_exit_evidence.py submit_formal.sh strict_metric_parser.py'.split())")
   manifest_line="manifest=loads(own_file(str(base/'tools/"+formal+"/source-manifest.json'),base).read_text())"
   pinned_loader="""def load_reviewed_formal_manifest(base):
 expected=os.environ.get('OPSD_STAGE_MANIFEST_SHA256','')
 assert re.fullmatch('[0-9a-f]{64}',expected), 'Externally reviewed final formal manifest SHA256 required'
 path=own_file(str(base/'tools/FORMAL_DIRECTORY/source-manifest.json'),base)
 assert digest(path)==expected, 'Formal manifest changed from externally reviewed final pin'
 return loads(path.read_text())
""".replace('FORMAL_DIRECTORY',formal)
   s=replace_once(s,'def verify_deployed(base=OWN):',pinned_loader+'def verify_deployed(base=OWN):')
   s=replace_once(s,manifest_line,'manifest=load_reviewed_formal_manifest(base)')

   old_metrics="""step=int(m[1]);steps.append(step);metrics={}
  for item in line[m.end():].split(' - '):
   key,sep,value=item.rpartition(':')
   if not sep:continue
   assert key not in metrics, ('Duplicate outer metric',key)
   value=float(value);assert math.isfinite(value), ('Nonfinite metric',key)
   metrics[key]=value"""
   new_metrics="""from strict_metric_parser import parse_metric_summary
  step,metrics,events=parse_metric_summary(line);steps.append(step)
  normalization_events.extend(events)"""
   s=replace_once(s,old_metrics,new_metrics)
   s=replace_once(s,"text=re.sub(r'\\x1b\\[[0-9;]*m','',text);steps=[]","steps=[];normalization_events=[]")
   s=replace_once(s,"'numeric_metrics_validated':True}","'numeric_metrics_validated':True,'recognized_trailing_ray_warning_frames':normalization_events}")
   s=replace_once(s,"verify_slurm(paths[stage+'_slurm'].read_text(),job)","""slurm_text=paths[stage+'_slurm'].read_text()
  if stage=='pilot' and slurm_text.lstrip().startswith('{'):
   from controller_exit_evidence import verify_controller_exit
   verify_controller_exit(loads(slurm_text),job,COMMIT,pilot,own_file,digest,loads)
  else:verify_slurm(slurm_text,job)""")
   s=replace_once(s,'return r\ndef load_reviewed_formal_manifest',f"assert r['picked_local_micro']=={micro} and r['picked_global_micro']=={8*micro}\n from micro_evidence import verify_micro_evidence\n verify_micro_evidence(review,pilot,r['pilot_job'],COMMIT,{micro})\n return r\ndef load_reviewed_formal_manifest")
   needle="return digest(base/'tools/"+formal+"/source-manifest.json')"
   extra="""assert set(manifest['pilot_additional_sources'])=={'pilot_audit.py','prior_cleanup.py','source_bundle_check.py'}
 for name,expected in manifest['pilot_additional_sources'].items():assert digest(own_file(str(base/'tools'/manifest['pilot_runtime_directory_name']/name),base))==expected
 assert set(manifest['hf_sources'])=={'reload_job.sh','checkpoint_reload.py','resource_sample.py','source_bundle_check.py'}
 for name,expected in manifest['hf_sources'].items():assert digest(own_file(str(base/'tools'/manifest['hf_runtime_directory_name']/name),base))==expected
 for field,folder in (('pilot_manifest_sha256','pilot_runtime_directory_name'),('hf_manifest_sha256','hf_runtime_directory_name')):assert digest(own_file(str(base/'tools'/manifest[folder]/'source-manifest.json'),base))==manifest[field]
 assert subprocess.check_output(['git','-C',str(base/'repo'),'rev-parse','HEAD'],text=True).strip()==COMMIT
 subprocess.run(['git','-C',str(base/'repo'),'diff','--quiet','HEAD'],check=True)
 assert set(manifest['production_math_sources'])=={'search_r1/baselines/actor.py','search_r1/baselines/distillation.py'}
 for name,expected in manifest['production_math_sources'].items():assert digest(own_file(str(base/'repo'/name),base))==expected
 """
   s=replace_once(s,needle,extra+needle)
  (out/n).write_text(s)
 for n in ('micro_evidence.py','collect_receipt.py','controller_exit_evidence.py','strict_metric_parser.py'):shutil.copy2(HERE/n,out/n)
 m=dict(source_commit=commit,status='formal_preview_only_actual_evidence_required',formal_directory_name=formal,pilot_runtime_directory_name=pm['directory_name'],hf_runtime_directory_name=hm['directory_name'],picked_local_micro=micro,picked_global_micro=8*micro,production_math_sources=p['production_math_sources'],runtime_sources=pm['runtime_sources'],pilot_additional_sources={n:v for n,v in pm['file_sources'].items() if n not in RUNTIME_NAMES},hf_sources=hm['file_sources'],pilot_manifest_sha256=sha(pilot_path),hf_manifest_sha256=sha(hf_path),formal_sources={n:sha(out/n) for n in ('formal_entry.sh','formal_job.sh','formal_supervisor.py','verify_formal_prerequisites.py','micro_evidence.py','collect_receipt.py','controller_exit_evidence.py','submit_formal.sh','strict_metric_parser.py')})
 (out/'source-manifest.json').write_text(json.dumps(m,indent=2)+'\n');return m
if __name__=='__main__':
 assert len(sys.argv)==3,'Filled actual formal profile and NEW local output directory'
 render(sys.argv[1],sys.argv[2])
