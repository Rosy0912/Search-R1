"""V5-only local review. All external operations denied before wrapper imports.

Only exact local Python startup fixtures explicitly authorized below may spawn.
No source/V4 suite rerun, remote.py, credentials, assets, tensors, scheduler/GPU.
"""
import ast,builtins,contextlib,hashlib,io,json,os,socket,subprocess,sys,tempfile,uuid
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
V5=BASE/'tree-fresh200-train-runtime-v5';V4=BASE/'tree-fresh200-train-runtime-v4'
V4REVIEW=BASE/'tree-fresh200-formal-v4-02b1-context-free-review'
SOURCE=BASE/'tree-behavior-mu-utf8-v2'
NONCE=uuid.uuid4().hex
attempted_external=[];allowed_children=[];allow=None
original_popen=subprocess.Popen;original_run=subprocess.run
def deny(*args,**kwargs):
    attempted_external.append(repr(args)[:200]);raise AssertionError('V5 CPU review external operation forbidden')
def permitted_popen(argv,*args,**kwargs):
    if allow is None or list(argv)!=allow['argv'] or kwargs.get('env')!=allow['env'] or kwargs.get('cwd')!=allow['cwd']:return deny(argv)
    allowed_children.append({'argv':list(argv),'scope':'exact_local_tiny_python_fixture','nonce':NONCE})
    return original_popen(argv,*args,**kwargs)
subprocess.Popen=permitted_popen
for name in ('call','run','check_call','check_output'):setattr(subprocess,name,deny)
socket.create_connection=deny
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo','os.exec','os.posix_spawn','os.system'):deny(event,args)
    if event=='subprocess.Popen':
        if allow is None or list(args[1])!=allow['argv'] or args[2]!=allow['cwd'] or args[3]!=allow['env']:deny(event,args)
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=os.fsdecode(args[0])
        if path.startswith('/dataset/') or path.endswith(('.pt','.safetensors')):raise AssertionError('V5 review forbids official/tensor asset reads')
sys.addaudithook(audit);sys.dont_write_bytecode=True
sys.path[:0]=[str(V5),str(SOURCE)]
os.environ.pop('TREE_INSTRUCT_TOOLS',None);os.environ.pop('TREE_FIXED_ACTOR_PROFILE',None)
os.environ['CUDA_VISIBLE_DEVICES']=''
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={
 'inventory':(V5/'FILE_HASHES_V5.json','5299a43d1f1615486fe3296ba04244ba03d553730c4f547b890403e6a7de2c8b'),
 'owner':(V5/'SOURCE_CPU_RECEIPT_V5.json','58ce3e6537aa4b07bc3c309606b09cfd7c7f3d4d2d577b988e0b82029ec95b44'),
 'patch':(V5/'V5_CHANGES.patch','269e3cb08e51361943f0a2aedc9c5dcde0d7d07dc8101ad8891226dfd3944b80'),
 'adapter':(V5/'SOURCE_INTERFACE_COMPATIBLE_RECEIPT.json','455fc4519b4baa36b8faf73946f993c87a5f34ed2c8d36f608b24c2c03d3a8bb'),
 'v4_reject':(V4REVIEW/'RECEIPT.json','4e618c6daee703dd84db9cbbb57ec87d8f0281287220aca85a166457d7194619'),
 'v4_exact_review':(V4REVIEW/'REVIEW.json','d39fb77d7be107317b08d8a56f0130eafcb18712f53ca5419b954d3a382b5bf7')}
binding={key:dict(path=str(path),expected_sha256=expected,actual_sha256=sha(path),matches=sha(path)==expected) for key,(path,expected) in pins.items()}
assert all(item['matches'] for item in binding.values())
inventory=json.loads(pins['inventory'][0].read_text());owner=json.loads(pins['owner'][0].read_text())
prior=json.loads((V4REVIEW/'EXACT_BYTE_AUDIT_RESULT.json').read_text())
wrappers={name:dict(path=str(V5/name),sha256=sha(V5/name),matches_inventory=sha(V5/name)==expected,
                    unchanged_from_v4=sha(V5/name)==prior['wrapper_records'][name]['sha256']) for name,expected in inventory['wrapper_files'].items()}
assert len(wrappers)==33 and all(x['matches_inventory'] for x in wrappers.values())
changed=sorted(name for name,item in wrappers.items() if not item['unchanged_from_v4'])
assert changed==['launch.sbatch','run_tree.sh','sitecustomize.py','slurm_identity.py']
assert sorted(owner['changed_wrapper_files'])==changed and owner['wrapper_files']==inventory['wrapper_files']
assert all(sha(V5/name)==expected for name,expected in inventory['files'].items())
sources={name:dict(path=item['path'],sha256=sha(item['path']),matches_prior_exact=sha(item['path'])==item['sha256']) for name,item in prior['source_records'].items()}
assert len(sources)==19 and all(x['matches_prior_exact'] for x in sources.values())
assert (V5/'SOURCE_INTERFACE_COMPATIBLE_RECEIPT.json').read_bytes()==(V4/'SOURCE_INTERFACE_COMPATIBLE_RECEIPT.json').read_bytes()
for key in ('actual_first5_accepted','actual_zero_accepted','actual_gpu_executed','actual_slurm_executed','actual_remote_executed','training_release','independent_dispatch_release','official_assets_loaded','model_weights_loaded'):
    assert owner[key] is False,key
for name,entry in inventory['runtime_entries'].items():
    assert entry=={'path':str(V5/name),'sha256':inventory['wrapper_files'][name]}
import build_package_draft as builder
prepared=builder.prepare_wrapper_inventory(inventory,{'runtime_entries':inventory['runtime_entries']},V5)
assert set(prepared)==set(wrappers)
exact=dict(input_bindings=binding,wrapper_count=33,wrapper_records=wrappers,source_closure_count=19,
           source_records=sources,changed_wrapper_count=4,changed_wrapper_files=changed,unchanged_wrapper_count=29,
           full_inventory_count=len(inventory['files']),all_pins_verified=True,
           current_runtime_entries=inventory['runtime_entries'],actual_builder_inventory_consumption_passed=True,
           source_cpu_mu_checker_unchanged=True,source_adapter_unchanged=True,
           counting_guards_same_instance_gate_unchanged=True,v4_reject_preserved=True)

def local_child(code,env,cwd):
    global allow
    argv=[sys.executable,'-B','-s','-c',code]
    allow={'argv':argv,'env':env,'cwd':str(cwd)}
    try:return original_run(argv,env=env,cwd=str(cwd),capture_output=True,text=True,timeout=10)
    finally:allow=None
startup=[]
for mode in ('import','install'):
    with tempfile.TemporaryDirectory(prefix='v5-independent-'+NONCE+'-') as tmp:
        root=Path(tmp);(root/'search_r1/baselines').mkdir(parents=True)
        (root/'search_r1/__init__.py').write_text('');(root/'search_r1/baselines/__init__.py').write_text('')
        fixture="raise RuntimeError('independent required import failure')\n" if mode=='import' else "import sys,types\ndef gpu_snapshot():pass\nclass Locked(types.ModuleType):\n def __setattr__(self,n,v):\n  if n=='gpu_snapshot':raise RuntimeError('independent required install failure')\n  super().__setattr__(n,v)\nsys.modules[__name__].__class__=Locked\n"
        (root/'search_r1/baselines/preflight.py').write_text(fixture)
        payload=root/('PAYLOAD_'+NONCE)
        env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(V5)+':'+str(root),'TREE_INSTRUCT_TOOLS':'required-independent-tiny-fixture',
             'TREE_FIXED_ACTOR_PROFILE':'0','CUDA_VISIBLE_DEVICES':'','PYTHONOPTIMIZE':'0','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}
        code=f"from pathlib import Path;Path({str(payload)!r}).write_text({NONCE!r})"
        result=local_child(code,env,root)
        assert result.returncode!=0 and not payload.exists()
        assert 'Required Tree own-cgroup GPU-query guard bootstrap failed' in result.stderr
        startup.append(dict(mode=mode,returncode=result.returncode,payload_ran=False,fatal_message_present=True,
                            stdout=result.stdout,stderr=result.stderr,actual_python_startup=True))

import slurm_identity as identity
from search_r1.baselines import preflight
OWN={'SLURM_JOB_ID':'99','SLURM_JOB_NODELIST':'h100-9','SLURM_CLUSTER_NAME':'wuhan-h100','CUDA_VISIBLE_DEVICES':'0'}
query_calls=[]
def query_sentinel(argv,*a,**kw):query_calls.append(list(argv));raise AssertionError('Unexpected query at mocked boundary')
invalid=[]
for cgroup in ('','0::/user.slice\n','0::/slurm/job_98/step_batch\n','0::/slurm/job_99/step_batch\n1:cpu:/slurm/job_98/step_batch\n'):
    with patch.dict(os.environ,OWN,clear=True),patch.object(identity.socket,'gethostname',return_value='h100-9'),patch.object(identity.Path,'read_text',return_value=cgroup),patch.object(subprocess,'check_output',side_effect=query_sentinel),patch.object(subprocess,'run',side_effect=query_sentinel):
        try:identity.guarded_preflight_main()
        except ValueError as error:invalid.append(dict(cgroup=cgroup,error=str(error),query_calls=0))
        else:raise AssertionError('Invalid actual cgroup accepted')
assert query_calls==[]
# Legal canonical entry preserves CLI, and its snapshot retains the current
# cgroup wrapper. Change actual cgroup immediately before a canonical snapshot.
valid=[]
for flags in ([],['--reuse-inherited']):
    argv=['-c',*flags,'--','childpython','owned.py','--nonce='+NONCE];calls=[]
    with patch.dict(os.environ,OWN,clear=True),patch.object(identity.socket,'gethostname',return_value='h100-9'),patch.object(identity.Path,'read_text',return_value='0::/slurm/job_99/step_batch\n'),patch.object(sys,'argv',argv),patch.object(preflight,'main',side_effect=lambda:calls.append(sys.argv.copy())):
        identity.guarded_preflight_main();assert calls==[argv]
    valid.append(dict(flags=flags,argv_exact=True,canonical_main_called_once=True))
with patch.dict(os.environ,OWN,clear=True),patch.object(identity.socket,'gethostname',return_value='h100-9'),patch.object(identity.Path,'read_text',return_value='0::/slurm/job_98/step_batch\n'),patch.object(subprocess,'check_output',side_effect=query_sentinel):
    try:preflight.gpu_snapshot()
    except ValueError:pass
    else:raise AssertionError('Canonical snapshot lost actual cgroup wrapper')
assert query_calls==[]
execs=[];argv=['-c','--reuse-inherited','--','childpython','owned.py','--nonce='+NONCE]
with patch.object(sys,'argv',argv),patch.dict(os.environ,{'SEARCH_R1_GPU_LEASES':'[]'}),patch.object(preflight,'validate_inherited_gpu_leases',return_value=None) as leases,patch.object(preflight.os,'execvp',side_effect=lambda *args:execs.append(args)),contextlib.redirect_stdout(io.StringIO()):
    preflight.main();leases.assert_called_once_with()
assert execs==[('childpython',['childpython','owned.py','--nonce='+NONCE])]
launch=(V5/'launch.sbatch').read_text();runner=(V5/'run_tree.sh').read_text()
expression='from slurm_identity import guarded_preflight_main; guarded_preflight_main()'
assert launch.count(expression)==1 and runner.count(expression)==2
assert "'-m','search_r1.baselines.preflight'" not in launch and 'python -m search_r1.baselines.preflight' not in runner
tree=ast.parse((V5/'slurm_identity.py').read_text());entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='guarded_preflight_main')
assert [ast.unparse(n) for n in entry.body[1:]]==['verify_runtime_cgroup()', 'install_gpu_query_guard()', 'from search_r1.baselines import preflight', 'preflight.main()']
result=dict(nonce=NONCE,exact_binding=exact,necessary_delta_checks_passed=True,
            actual_fatal_python_startup_checks=startup,current_cgroup_invalid_cases=invalid,
            legal_canonical_cli_checks=valid,canonical_snapshot_rechecks_current_cgroup=True,
            original_reuse_lease_and_child_exec_preserved=True,all3_formal_entries_ast_verified=True,
            local_tiny_python_children_count=len(allowed_children),local_children_exact_argv_environment_allowlist=True,
            all_other_subprocess_network_route_scheduler_gpu_exec_paths_failclosed=True,
            unexpected_external_operations=attempted_external,actual_gpu_queries=0,actual_scheduler_queries=0,
            actual_remote_calls=0,official_asset_loads=0,pt_deserializations=0,training_outers=0,optimizer_writes=0,
            source_tests_rerun=False,v4_9_tests_rerun=False,full_suite_rerun=False,gloo_rerun=False)
assert not attempted_external
with (HERE/'INDEPENDENT_V5_DELTA_RESULT.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(necessary_delta_pass=True,pinned_inputs=len(binding),wrapper_count=33,source_count=19,
                      changed_wrapper_count=4,actual_tiny_startup_children=2,actual_external_gpu_scheduler_calls=0,nonce=NONCE)))
