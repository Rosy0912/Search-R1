"""Independent V4 review probes: all subprocess/network/GPU paths fail closed.

Only local metadata and synthetic CPU fixtures are loaded. No remote.py import.
"""
import contextlib
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import socket
import site
import subprocess
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
V4=BASE/'tree-fresh200-train-runtime-v4'
SOURCE=BASE/'tree-behavior-mu-utf8-v2'
blocked=[]
def deny(*args,**kwargs):
    blocked.append(repr(args)[:240])
    raise AssertionError('Independent CPU review forbids subprocess/network/exec')
for name in ('Popen','call','run','check_call','check_output'):
    setattr(subprocess,name,deny)
socket.create_connection=deny
def audit(event,args):
    if event in ('subprocess.Popen','socket.connect','socket.getaddrinfo','os.exec','os.posix_spawn','os.system'):
        deny(event,args)
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=os.fsdecode(args[0])
        if path.startswith('/dataset/') or path.endswith(('.pt','.safetensors')):
            raise AssertionError('Independent CPU review forbids official/tensor asset reads')
sys.addaudithook(audit)
sys.dont_write_bytecode=True
# -I skips user site initialization; expose existing CPU dependencies only after
# failclosed controls have been installed. No dependency installation occurs.
sys.path.append('/Users/shaoshitong/Library/Python/3.9/lib/python/site-packages')
sys.path[:0]=[str(V4),str(SOURCE)]
os.environ.pop('TREE_INSTRUCT_TOOLS',None)
os.environ.pop('TREE_FIXED_ACTOR_PROFILE',None)
os.environ['CUDA_VISIBLE_DEVICES']=''

import test_production_v4_cpu as cases
import slurm_identity as identity
import search_r1.baselines.preflight as preflight

results={'subprocess_network_failclosed_installed_before_runtime_imports':True,
         'official_asset_reads':0,'pt_deserializations':0,'real_gpu_queries':0,
         'real_scheduler_queries':0,'remote_calls':0,'training_outers':0,
         'optimizer_writes':0,'full_suite_rerun':False,'gloo_rerun':False}
names=[name for name in unittest.defaultTestLoader.getTestCaseNames(cases.V4Tests)
       if name!='test_new_cpu_checker_fresh_process_real_registration_and_semantic_drift']
suite=unittest.TestSuite(cases.V4Tests(name) for name in names)
log=io.StringIO()
run=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
results['changed_interface_tests']={'methods':names,'count':run.testsRun,'passed':run.wasSuccessful(),
                                  'failures':len(run.failures),'errors':len(run.errors)}

# This process itself starts independently, with no parent tokenizer registry.
# Use the real saved-data checker registration and real tiny source validator.
import search_r1.baselines.tree_raw_tokens as raw
from test_tree_behavior_mu_data_cpu import BehaviorMuDataTests
import check_behavior_mu_cpu as checker
import torch
torch.load=deny
raw._VALIDATION_TOKENIZERS.clear()
fixture=BehaviorMuDataTests();fixture.setUp()
try:
    data=fixture.output();raw._VALIDATION_TOKENIZERS.clear()
    try:fixture.validate(data)
    except ValueError as error:missing=str(error)
    else:raise AssertionError('Fresh process missing registry accepted')
    path=Path(fixture.tmp.name).resolve()
    for name in checker.TOKENIZER_FILES:
        if not (path/name).exists():(path/name).write_bytes(b'{"independent_tiny_fixture":true}')
    package={'official_model_path':str(path),'model_identity':{'files':{
        name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in checker.TOKENIZER_FILES}}}
    utils=ModuleType('verl.utils');loads=[]
    def tiny_loader(value,**kwargs):loads.append((value,kwargs));return fixture.tok
    utils.hf_tokenizer=tiny_loader
    with patch.dict(sys.modules,{'verl.utils':utils}),patch.object(checker,'OFFICIAL_MODEL_PATH',path),patch.object(checker.socket,'gethostname',return_value='h100-9'),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
        tokenizer,binding=checker.initialize_validation_tokenizer(package)
        valid=fixture.validate(data)
        assert raw._validation_tokenizer(binding) is tokenizer
        fixture.tok.parts[2]='WRONG'
        try:fixture.validate(data)
        except ValueError as error:drift=str(error)
        else:raise AssertionError('Live backend semantic drift accepted')
    results['independent_current_process_mu_registration']={
        'passed':True,'process_registry_empty_before_tiny_fixture':True,
        'new_child_spawn_used':False,'actual_checker_registration_executed':True,
        'actual_tiny_mu_validator_executed':True,'missing_registry_error':missing,
        'backend_drift_error':drift,'tiny_loader_calls':loads,
        'official_tokenizer_executed':False,'weights_loaded':False}
finally:fixture.tearDown()

# The production launch runs -m preflight. Its fresh __main__ definitions must
# carry the guard installed on the canonical imported module; V4 does not.
gpu_sentinels=[]
def gpu_request_sentinel(argv,**kwargs):
    gpu_sentinels.append(list(argv))
    raise AssertionError('Intercepted GPU query before actual execution')
identity.install_gpu_query_guard()
with patch.object(identity.socket,'gethostname',return_value='wrong-host'),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'},clear=True),patch.object(subprocess,'check_output',side_effect=gpu_request_sentinel):
    try:preflight.gpu_snapshot()
    except ValueError as error:canonical_error=str(error)
    else:raise AssertionError('Canonical guard failed to reject wrong host')
    assert not gpu_sentinels
    stderr=io.StringIO()
    with patch.object(sys,'argv',['preflight','--','unreachable-command']),contextlib.redirect_stderr(stderr):
        try:runpy.run_module('search_r1.baselines.preflight',run_name='__main__')
        except AssertionError as error:main_error=str(error)
        else:raise AssertionError('Expected failclosed sentinel was not reached')
results['module_main_guard_bypass']={
    'reproduced':bool(gpu_sentinels),'canonical_wrong_host_refused_before_query':True,
    'canonical_error':canonical_error,'main_error':main_error,
    'intercepted_gpu_request_argv':gpu_sentinels,'actual_gpu_queries':0,
    'runpy_warning':stderr.getvalue()}
# Exercise Python's actual startup handler against the actual V4 sitecustomize
# file. Its required import fails locally; the handler returns and continues.
original_import=builtins.__import__
def failed_required_import(name,*args,**kwargs):
    if name=='slurm_identity':raise ValueError('Independent required-guard installation failure')
    return original_import(name,*args,**kwargs)
sys.modules.pop('sitecustomize',None)
startup_stderr=io.StringIO()
with patch.dict(os.environ,{'TREE_INSTRUCT_TOOLS':'independent-tiny-required-guard'},clear=True),patch.object(builtins,'__import__',side_effect=failed_required_import),contextlib.redirect_stderr(startup_stderr):
    site.execsitecustomize()
    startup_continued=True
results['startup_installation_exception_failopen']={
    'reproduced':startup_continued,'actual_stdlib_execsitecustomize_used':True,
    'actual_v4_sitecustomize_file_used':True,'required_guard_failure_suppressed':True,
    'stderr':startup_stderr.getvalue(),'real_gpu_scheduler_remote_calls':0}
results['blocked_unexpected_operations']=blocked
(HERE/'CPU_REVIEW_PROBES.log').write_text(log.getvalue())
(HERE/'CPU_REVIEW_PROBES_RESULT.json').write_text(json.dumps(results,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'changed_cpu_tests_passed':run.wasSuccessful(),'changed_cpu_tests_count':run.testsRun,
                  'tiny_mu_registration_passed':True,'module_main_guard_bypass_reproduced':bool(gpu_sentinels),
                  'actual_gpu_scheduler_remote_calls':0,'blocked_unexpected_operations':len(blocked)}))
