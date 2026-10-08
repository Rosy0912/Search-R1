import json,subprocess,shlex,sys,hashlib,re,datetime,importlib.util
from pathlib import Path
root='/mnt/data/shaoshitong/search-r1-rl/opsd'
bundle='formal-8e60976d364a-pilot-2994-hf-2996-reviewed-c85d38bd29f6'
code=f'''from pathlib import Path
import json,hashlib,sys,re,subprocess,datetime
root=Path({root!r});run=root/'runs/opsd-formal-3007';bundle=root/'tools'/{bundle!r}
sys.path.insert(0,str(bundle))
from verify_formal_prerequisites import verify_outers
b=(run/'training.log').read_bytes();s=b.decode()
steps=[int(m.group(1)) for line in s.splitlines() if 'actor/optimizer_steps:' in line for m in [re.search(r'(?:^|\\s)step:(\\d+) - ',line)] if m]
assert steps and steps[-1]>=50
v=verify_outers(s,steps[-1])
commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()
assert commit=='8e60976d364ab7d7ae51714415813c1359dafe9a'
subprocess.run(['git','-C',str(root/'repo'),'diff','--quiet','HEAD'],check=True)
print(json.dumps(dict(sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id='3007',source_commit=commit,tracked_clean=True,original_log_bytes=len(b),original_log_sha256=hashlib.sha256(b).hexdigest(),training_log=s,native_numeric_result=v,formal_manifest_sha256=hashlib.sha256((bundle/'source-manifest.json').read_bytes()).hexdigest(),strict_parser_sha256=hashlib.sha256((bundle/'strict_metric_parser.py').read_bytes()).hexdigest(),scontrol=subprocess.check_output(['scontrol','show','job','3007'],text=True),training_finished=(run/'training-finished.txt').exists())))
'''
r=subprocess.run(['python3','work/remote.py','env CUDA_VISIBLE_DEVICES= '+root+'/env/bin/python -c '+shlex.quote(code)],capture_output=True,text=True)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
x=json.loads(r.stdout)
e=Path('work/opsd/reproduction/baselines/opsd-validation-20261003/evidence')
(e/'job3007_actual50_reviewer_raw.json').write_text(r.stdout)
assert x['formal_manifest_sha256']=='eb13b32b6dbcd608c237805bc950023a627ea4bdd4e98596a1afd5c9417d7051'
assert x['strict_parser_sha256']=='6a7d064e544716ce6981f6c02a15e94ac46b8dd4e3e5c39bc203b85763e3b95a'
assert hashlib.sha256(x['training_log'].encode()).hexdigest()==x['original_log_sha256']
sys.path.insert(0,str(Path('work/formal-preparation/hf500-formal3007-strict-two-raywarnings-preview').resolve()))
from periodic_validation import verify_periodic_validations
from strict_metric_parser import parse_metric_summary
v=verify_periodic_validations(x['training_log'],expected_steps=(0,50),require_final=False)
clean=re.sub(r'\x1b\[[0-9;]*m','',x['training_log'])
j=[json.loads(line.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]) for line in clean.splitlines() if 'BASELINE_EVALUATION_METRICS_JSON=' in line]
assert len(j)==2
summaries=[]
for line in x['training_log'].splitlines():
 if re.search(r'(?:^|\s)step:(\d+) - ',line):
  step,metrics,events=parse_metric_summary(line)
  if step in (0,50):summaries.append(dict(step=step,metrics=metrics,recognized_warning_frames=events))
a=dict(sample_utc=x['sample_utc'],job_id='3007',source_commit=x['source_commit'],formal_manifest_sha256=x['formal_manifest_sha256'],original_log_bytes=x['original_log_bytes'],original_log_sha256=x['original_log_sha256'],native_numeric_result=x['native_numeric_result'],periodic_pair_result=v,original_validation_metrics=j,stage_summaries=summaries,score_protocol='Frozen 8e historical scoring protocol with reported scoring defect; not corrected efficacy',independent_numeric_and_periodic_acceptance=True,formal500_complete=False,final51713_complete=False,corrected_efficacy_complete=False)
(e/'job3007_actual50_independent_analysis.json').write_text(json.dumps(a,indent=2))
print(json.dumps({k:a[k] for k in ['sample_utc','original_log_bytes','original_log_sha256','native_numeric_result','periodic_pair_result']},indent=2))
for step,metrics in zip((0,50),j):
 print(step,{k:metrics[k] for k in metrics if k.endswith('/all') or k.startswith('val/qa_count/')})
for q in summaries:
 if q['step']==50:print('step50timings',{k:v for k,v in q['metrics'].items() if k.startswith('timing_s/')})
