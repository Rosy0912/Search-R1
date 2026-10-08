from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
h=lambda b:hashlib.sha256(b).hexdigest()
raw=(OUT/'decoded-source-pins.json').read_bytes();d=json.loads(raw)
assert d['torch_cuda_initialized_before'] is False and d['torch_cuda_initialized_after'] is False
assert d['CUDA_VISIBLE_DEVICES']=='' and d['hostname']=='h100-10'
assert d['versions']=={'torch':'2.4.0','transformers':'4.46.3','flash-attn':'2.7.4.post1'}
checks=[]
for k,p in d['files'].items():
 b=Path(p['local_path']).read_bytes();assert h(b)==p['sha256'] and len(b)==p['bytes'] and p['stat']==p['stat_after'];checks.append('stable_full_source_bytes:'+k)
nt_raw=(ROOT/'native-source-inventory-b611864d3f6a-ACTUAL.json').read_bytes();nt=json.loads(nt_raw)
assert h(nt_raw)=='ed7b1413a1dc38f77d4aa62999c9363139a6c92ef6c2358ed5455db169be398f'
ctx=nt['modules']['torch.utils._contextlib'];assert ctx['path']==d['files']['torch_contextlib']['path'] and ctx['sha256']==d['files']['torch_contextlib']['sha256'];checks.append('native_211_contextlib_pin_matches')
for k in ('hf_flash_utils','hf_qwen2','flash_interface'):
 assert d['files'][k]['path'] not in {x['path'] for x in nt['modules'].values()}
checks.append('new_HF_and_Flash_source_pins_not_retroactively_in_211')
def node(role,name,cls=None):
 tree=ast.parse(Path(d['files'][role]['local_path']).read_bytes())
 if cls:tree=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
 return next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
def default(n,key):
 return dict(zip([a.arg for a in n.args.args][-len(n.args.defaults):],n.args.defaults))[key]
hf=node('hf_flash_utils','_flash_attention_forward');assert isinstance(default(hf,'deterministic'),ast.Constant) and default(hf,'deterministic').value is None
checks.append('HF_deterministic_parameter_default_None')
getenv=[x for x in ast.walk(hf) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='get' and ast.unparse(x.func.value)=='os.environ' and x.args and isinstance(x.args[0],ast.Constant) and x.args[0].value=='FLASH_ATTENTION_DETERMINISTIC'];assert len(getenv)==1 and getenv[0].args[1].value=='0'
assert 'is_flash_attn_greater_or_equal("2.4.1")' in Path(d['files']['hf_flash_utils']['local_path']).read_text()
checks.append('HF_reads_switch_inside_each_call_for_native_supported_version')
qw=node('hf_qwen2','forward','Qwen2FlashAttention2');calls=[x for x in ast.walk(qw) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='_flash_attention_forward'];assert len(calls)==1 and 'deterministic' not in {kw.arg for kw in calls[0].keywords};checks.append('Qwen2_production_call_uses_helper_default')
for f in ('flash_attn_func','flash_attn_varlen_func'):
 fn=node('flash_interface',f);assert default(fn,'deterministic').value is False;checks.append(f+'_default_false')
flash=Path(d['files']['flash_interface']['local_path']).read_text();assert 'ctx.deterministic = deterministic' in flash and 'ctx.deterministic,' in flash and 'flash_attn_gpu.bwd(' in flash and 'flash_attn_gpu.varlen_bwd(' in flash
checks.append('backward_deterministic_forwarded_to_native_kernel')
assert 'FLASH_ATTENTION_DETERMINISTIC' not in flash and 'are_deterministic_algorithms_enabled' not in flash and 'are_deterministic_algorithms_enabled' not in Path(d['files']['hf_flash_utils']['local_path']).read_text();checks.append('no_automatic_torch_global_flag_translation_in_these_interfaces')
b=json.loads((ROOT/'runtime-tools-bound-b611864d3f6a-v3.json').read_bytes());repo=ROOT.parent/'opsd-unclipped-multitraj200-nativehead';assert b['source_commit']=='b611864d3f6ad048189b11b9b9613854e789da5d'
for f,pin in b['source_files'].items():assert h((repo/f).read_bytes())==pin
checks.append('all_19_actual_bound_project_sources_match')
for name in ('search_r1.baselines.distillation','verl.workers.fsdp_workers'):
 rec=nt['modules'][name];rel=rec['path'].split('/repo-b611864d3f6a/')[1];assert rec['sha256']==b['source_files'][rel]
checks.append('production_attention_loaders_join_native_211_and_source19')
dist=(repo/'search_r1/baselines/distillation.py').read_text();worker=(repo/'verl/workers/fsdp_workers.py').read_text();probe=(repo/'reproduction/baselines/e1-multitraj200/microbatch_probe.py').read_text();assert "attn_implementation='eager'" in dist and "attn_implementation='flash_attention_2'" in worker and "attn_implementation='flash_attention_2'" in probe
checks.append('teacher_eager_student_FA2_paths_proven')
print(json.dumps({'status':'PASS_CPU_STATIC_SOURCE_CONTRACT_NOT_GPU_REPEATABILITY','checks':checks,'count':len(checks),'native_inventory_sha256':h(nt_raw),'decoded_pins_sha256':h(raw)},indent=2))
