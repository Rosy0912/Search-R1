from pathlib import Path
import ast,base64,hashlib,json,types
H=Path(__file__).resolve().parents[2];C=H/'checkpoint50-observation-candidate'
pins={'collect_checkpoints50_100.py':'e459fb4961183b48226e20ebe7ae4595a92ec27ac8b569a76eb5befc58bc2b9f','collect_checkpoints50_100_v2.py':'61368d54b0e6732c9fef424b18df1903cce3bd50ed2ba3582c8c122b8426deba','test_native_repeat_frame_delta.py':'3e2d27da9ef8ede8f7c6a9e8212c2a7d33e9805d2c22e3fcfcc5ea56d581317b','SOFTWARE_READINESS_V2.json':'feff48a9564cd0d5b6ed895154f26cfcb505cfe6f7ad6277ef55d3faaa785857'}
for name,pin in pins.items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==pin
old=ast.parse((C/'collect_checkpoints50_100.py').read_bytes());new=ast.parse((C/'collect_checkpoints50_100_v2.py').read_bytes())
def nodes(tree):return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=nodes(old),nodes(new);assert set(b)-set(a)=={'native_repeat_frame'} and {k for k in a if a[k]!=b[k]}=={'parse_events','collect'}
assert [ast.dump(n,include_attributes=False) for n in old.body if not isinstance(n,ast.FunctionDef)]==[ast.dump(n,include_attributes=False) for n in new.body if not isinstance(n,ast.FunctionDef)]
c=types.ModuleType('independent_native_repeat_v2');c.__file__=str(C/'collect_checkpoints50_100_v2.py');exec(compile((C/'collect_checkpoints50_100_v2.py').read_bytes(),c.__file__,'exec'),c.__dict__)
gate_bytes=(H/'job3304-first5-complete-originals-v3/raw/source/search_r1/baselines/checkpoint_gate.py').read_bytes();gate=c.load_gate(gate_bytes)
warning=(H/'job3304-first5-complete-originals-v3/raw/native-source/fully_sharded_data_parallel.py').read_bytes();assert c.sha(warning)==c.FSDP_SHA
proofs=[]
for off in (4866679,4923202):
 raw=(H/'checkpoint50_100-collection-once-v1-original'/f'failed-frame-{off}.original').read_bytes();events,partial=c.parse_events(raw,gate,warning)
 assert len(events)==1 and partial==[];e=events[0];proof=e['native_repeat_transport'];assert base64.b64decode(e['raw_base64'],validate=True)==raw and c.sha(raw)==e['sha256']
 start,length=proof['tag_offset_in_line'],proof['tag_bytes'];tag=raw[start:start+length];assert c.sha(tag)==proof['tag_sha256'] and base64.b64decode(proof['tag_base64'],validate=True)==tag
 assert c.sha(raw[start+length:])==proof['remaining_suffix_sha256'] and len(raw[start+length:])==proof['remaining_suffix_bytes']
 checked=gate.parse_log_marker((raw[:start]+raw[start+length:]).decode(),e['marker'],driver_pid=c.DRIVER[0],job_id='3304',worker_pids=tuple(x[0] for x in c.WORKERS.values()),checkpoint_warning_path=c.TORCH_ROOT/'utils/checkpoint.py',fsdp_warning_source_bytes=warning)
 assert c.canonical(checked)==c.canonical(e['value']);proofs.append({'original_offset':off,'raw_sha256':e['sha256'],'proof':proof})
# Narrow independent rejection composition not in the author's5 methods: native JSON decoder errors never turn into a successful fallback.
raw=(H/'checkpoint50_100-collection-once-v1-original/failed-frame-4866679.original').read_bytes()
for bad in (raw.replace(b'"version": 3',b'"version": 3, "version": 3',1),raw.replace(b'"version": 3',b'"version": NaN',1)):
 assert bad!=raw
 try:c.parse_events(bad,gate,warning)
 except ValueError:pass
 else:raise AssertionError('Strict decoder boundary was weakened')
ready=json.loads((C/'SOFTWARE_READINESS_V2.json').read_bytes());ex=ready['cpu_originals'][0];p=Path(ex['path']);bb=p.read_bytes();assert hashlib.sha256(bb).hexdigest()==ex['sha256'];execution=json.loads(bb)
assert execution['waited'] is True and execution['returncode']==0 and execution['inputs_unchanged'] is True and ready['actual_tests']==5 and ready['actual_waited'] is True and ready['actual_returncode']==0
for item in ready['cpu_originals'][1:]:
 raw=Path(item['path']).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
for name,pin in pins.items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==pin
print(json.dumps({'status':'PASS_TWO_ACTUAL_FRAME_LOCAL_COMPOSITION_AND_STRICT_DECODER_BOUNDARIES','proofs':proofs,'author5_truewait0_originals_audited_not_rerun':True,'all_legacy_functions_except_parse_events_collect_ast_equal':True,'nonfunction_AST_equal':True,'runtime_sha256':pins['collect_checkpoints50_100_v2.py'],'production_main':False,'remote':False,'gpu':False,'pt_hash_or_load':False},sort_keys=True))
