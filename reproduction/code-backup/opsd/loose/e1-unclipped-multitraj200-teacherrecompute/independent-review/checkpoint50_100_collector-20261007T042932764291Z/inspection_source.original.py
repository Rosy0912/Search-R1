from pathlib import Path
import base64,builtins,dis,hashlib,json,math,types
H=Path(__file__).resolve().parents[2]
C=H/'checkpoint50-observation-candidate'
expected={'collect_checkpoints50_100.py':'e459fb4961183b48226e20ebe7ae4595a92ec27ac8b569a76eb5befc58bc2b9f','test_checkpoint_observation.py':'addf050e75796780db0ae1795690b45409fb5bf76822d22772ee4397727723b9','SOFTWARE_READINESS.json':'392c76f2af6a5c389981296c06dc27b5543bda1208c3d3566dd226e5815eacdf','DESIGN.md':'412e62ecf5b45dcb0c8d61f81d63587636adc68066507deeb3d9a9427080c198'}
refs={}
for name,pin in expected.items():
 p=C/name;raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==pin;refs[name]={'sha256':pin,'bytes':len(raw)}
raw=(C/'collect_checkpoints50_100.py').read_bytes();module=types.ModuleType('independent_checkpoint50_100');module.__file__=str(C/'collect_checkpoints50_100.py');code=compile(raw,module.__file__,'exec',dont_inherit=True,optimize=0);exec(code,module.__dict__)
missing=[]
def recurse(value):
 for item in dis.get_instructions(value):
  if item.opname=='LOAD_GLOBAL' and item.argval not in module.__dict__ and item.argval not in vars(builtins):missing.append((value.co_name,item.argval))
 for child in value.co_consts:
  if isinstance(child,types.CodeType):recurse(child)
recurse(code);assert not missing
assert module.SOURCE_FILES==json.loads((C/'source-pins.json').read_bytes())
for relative,pin in module.SOURCE_FILES.items():assert hashlib.sha256((H.parent/'opsd-unclipped-multitraj200-teacherrecompute'/relative).read_bytes()).hexdigest()==pin
packet=json.loads((H/'job3304-release-live-originals-v15/transport.stdout.original').read_bytes());assert (packet['driver_pid'],int(packet['driver_start']))==module.DRIVER
assert len({p for p,s in module.WORKERS.values()})==8 and module.DRIVER[0] not in {p for p,s in module.WORKERS.values()}
for rank,(pid,start) in module.WORKERS.items():assert int(packet['workers'][str(pid)]['stat'].rsplit(')',1)[1].split()[19])==start
assert hashlib.sha256((H/'job3304-release-live-originals-v15/acceptance-5-receipt.json').read_bytes()).hexdigest()==module.RECEIPT
original=H/'checkpoint50-observation-candidate/cpu-originals/20261007T042008145422Z';execution=json.loads((original/'execution.original.json').read_bytes())
assert hashlib.sha256((original/'execution.original.json').read_bytes()).hexdigest()=='9189116e2cb5619322bc1c60195d5b647dd86ea5fa266a8f652a9206bb94862c'
assert execution['waited'] is True and execution['returncode']==0 and execution['tests']==6 and execution['inputs_stable'] is True
for key in ('stdout.original','stderr.original'):
 raw=(original/key).read_bytes();assert execution[key]['sha256']==hashlib.sha256(raw).hexdigest() and execution[key]['bytes']==len(raw)
assert b'Ran 6 tests' in (original/'stderr.original').read_bytes() and b'OK' in (original/'stderr.original').read_bytes()
driver_path=H/'job3304-release-live-originals-v15/driver-current-prefix.out';raw=driver_path.read_bytes();evaluations=[];outers=[]
for line in raw.decode().splitlines():
 if 'BASELINE_EVALUATION_METRICS_JSON=' in line:
  value,end=json.JSONDecoder().raw_decode(line.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1].lstrip());evaluations.append(value)
 if 'OPSD_E1_OUTER_JSON=' in line:
  value,end=json.JSONDecoder().raw_decode(line.split('OPSD_E1_OUTER_JSON=',1)[1].lstrip());outers.append(value)
assert len(outers)==111 and [v['actual_outer_updates'] for v in outers]==list(range(1,112))
summary=[]
for value,step,cohort in zip(evaluations,[0,5,50,50,100,100],['dev512','stage64']*3):
 count=512 if cohort=='dev512' else64
 assert value['val/qa_count/all']==count
 assert {key[len('val/qa_count/'):]:value[key] for key in value if key.startswith('val/qa_count/') and not key.endswith('/all')}=={'nq':count//2,'hotpotqa':count//2}
 em_count=0
 for name in ('nq','hotpotqa'):
  n=value['val/qa_count/'+name];em=value['val/em/'+name]*n;assert math.isclose(em,round(em),abs_tol=1e-12);em_count+=round(em)
  assert value['val/evaluated_count/'+name]==n
  assert math.isclose(value['val/test_score/'+name],value['val/em/'+name],abs_tol=1e-12)
 for metric in ('em','f1','answer_format','format_rate','generated_tokens','search_calls'):
  weighted=sum(value['val/'+metric+'/'+name]*value['val/qa_count/'+name] for name in ('nq','hotpotqa'))/count
  assert math.isclose(value['val/'+metric+'/all'],weighted,rel_tol=1e-12,abs_tol=1e-12)
 assert value['val/em/all']==em_count/count
 if step in (50,100):
  outer=outers[step-1]['metrics'];prefix='val/' if cohort=='dev512' else 'opsd_stage/'
  for key,number in value.items():assert outer[key if prefix=='val/' else prefix+key]==number
 summary.append({'step':step,'cohort':cohort,'count':count,'em_count':em_count,'em':value['val/em/all'],'f1':value['val/f1/all']})
for name,pin in expected.items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==pin
print(json.dumps({'scope':'INDEPENDENT_SOFTWARE_STATIC_AUTHOR6_ORIGINALS_AUDIT_AND_EXISTING111_LOG_ARITHMETIC_NOT_REMOTE_COLLECTION_OR_NATIVE_RELOAD','inputs':refs,'recursive_namespace_missing':missing,'source19_verified':True,'actual_v15_unique_driver8worker_pid_start_match':True,'author6_originals_truewait0_audited_not_rerun':True,'existing_log_arithmetic':summary,'production_main':False,'remote':False,'gpu':False,'native_pt_read':False},sort_keys=True))
