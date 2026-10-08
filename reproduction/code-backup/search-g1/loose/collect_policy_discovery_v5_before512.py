"""Readonly existing before512 arithmetic. No new evaluation or default RPC.

Reuse the immutable v3 diagnostic's original FAILED/source/512 readers. Native
CPU reads only source, small metadata and the fixed512 two-column shared product;
raw QA remains native. No tokenizer, model, optimizer, token journal or kernels.
"""
import argparse
import importlib.util
import inspect
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
HELPER_SHA256='1c00e61981f8a37daf8674126296ff1ff1953ace8ad7dc0e08e696d4e1106c13'


def helper():
    path=HERE/'diagnose_policy_discovery_v5_special_tokens_v3.py'
    spec=importlib.util.spec_from_file_location('_immutable_failed3294_v3',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.require(module.sha(path.read_bytes())==HELPER_SHA256,'Immutable v3 512 reader changed')
    return module


def native_before512(payload,reader):
    started=time.time();launch=payload['launch'];terminal=payload['terminal'];base=Path(launch['base'])
    require(started>=terminal['observation_finished_unix'],'512 read precedes original failed capture')
    route=reader.current_completed_terminal(terminal)
    authentic_sources(base,launch['source_sha256'])
    raw,receipt=readonly_bytes(base/'logs/run-manifest.json',4*1024**2)
    run=strict_json(raw)
    require(run['run_id']==RUN_ID and run['parent_production_run_id']==PARENT_RUN_ID and
        run['policy_calibration_slurm_job_id']==launch['job_id']=='3294' and run['source_commit']==SOURCE_COMMIT and
        run['source_sha256']==launch['source_sha256'] and run['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1' and
        type(run['completed_updates']) is int and run['completed_updates']==0,
        'Original failed run/source/official revision/actual completed count differs')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    sys.path.insert(0,str(base/'repo'))
    before=before512_diagnosis(base,run,launch['source_sha256'])
    torch_module=sys.modules.get('torch')
    require(not(torch_module and torch_module.cuda.is_initialized()),'Readonly512 unexpectedly initialized CUDA')
    result=dict(role='failed3294_existing_before512_readonly_cpu',run_id=run['run_id'],slurm_job_id=launch['job_id'],
        parent_production_run_id=run['parent_production_run_id'],source_commit=SOURCE_COMMIT,
        source_map_sha256=sha(canonical(launch['source_sha256'])),
        source_snapshot_kind='current_disk_hashes_not_loaded_module_proof',
        original_failed_terminal_sha256=terminal['observation_sha256'],current_failed_terminal_route=route,
        original_manifest=receipt,existing_before512=before,completed_updates_at_original_manifest=run['completed_updates'],
        checked_unix=started,observation_finished_unix=time.time(),cuda_visible_devices_disabled=True,
        accepted=False,context_verified=False,production_admission=False,healthy_checkpoint_accepted=False,
        physical_execution_authenticated=False,health_evaluated=False,new_evaluation_generated=False,
        optimizer_writes_authenticated=False,model_or_adam_tensors_read=False,token_journal_read=False,
        raw_text_or_ids_or_lp_values_published=False)
    require(len(canonical(result))<=MAX_RETURN_BYTES,'Existing512 scalar projection exceeds bound')
    return result


def _literal(value):
    if isinstance(value,(set,frozenset)):
        return 'frozenset('+repr(tuple(sorted(value)))+')'
    return repr(value)


def remote_code(module,reader):
    prefix='import ast,datetime,hashlib,io,json,math,os,re,stat,subprocess,sys,time\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom collections import Counter\nfrom collections.abc import Mapping,Sequence\n'
    body='\n'.join(inspect.getsource(getattr(reader,name)) for name in
        ('canonical','sha','require','strict_json','_digest','_number','_job_fields','verify_terminal_current','current_completed_terminal'))
    body+='\ncommon=SimpleNamespace(require=require,sha=sha,canonical=canonical,strict_json=strict_json)\n'
    for name in ('INVALID_JOB','BIRTH_FIELDS'):body+=name+'='+_literal(getattr(reader,name))+'\n'
    code=prefix+'reader_ns={};exec('+repr(prefix+body)+',reader_ns);reader=SimpleNamespace(**reader_ns)\n'
    for name in ('RUN_ID','PARENT_RUN_ID','SOURCE_COMMIT','MAX_RETURN_BYTES','EMPTY_SOURCE_FILES','EMPTY_SOURCE_SHA256'):
        code+=name+'='+_literal(getattr(module,name))+'\n'
    code+='\n'.join(inspect.getsource(getattr(module,name)) for name in
        ('require','sha','canonical','strict_json','stable','readonly_bytes','authentic_sources',
         'evaluation_aggregate','before512_projection','before512_diagnosis'))
    code+=inspect.getsource(native_before512)
    code+='\np=json.load(sys.stdin);print(json.dumps(native_before512(p,reader),allow_nan=False))\n'
    compile(code,'<existing512-readonly>','exec');return code


def execute_once(artifacts,output,*,execute=False,rpc=None):
    module=helper();module.require(execute is True,'Explicit future execution required; no default RPC')
    reader,launch,terminal=module.input_contract(artifacts)
    output=Path(output)
    module.require(output.is_absolute() and output.parent.resolve()==output.parent and not output.exists() and
        not output.is_symlink(),'New exclusive absolute local output required')
    code=remote_code(module,reader);payload=dict(launch=launch,terminal=terminal)
    if rpc is None:
        sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
        def rpc(code,payload):
            command=shlex.quote(str(Path(launch['base'])/'env/bin/python'))+' -B -c '+shlex.quote(code)
            result=subprocess.run(control.route('h100-8',command),input=module.canonical(payload),capture_output=True,timeout=90)
            module.require(result.returncode==0,'Readonly512 failed; private stderr SHA '+module.sha(result.stderr))
            module.require(len(result.stdout)<=module.MAX_RETURN_BYTES,'Unexpected readonly512 output size')
            return module.strict_json(result.stdout)
    result=rpc(code,payload)
    module.require(result['role']=='failed3294_existing_before512_readonly_cpu' and result['run_id']==module.RUN_ID and
        result['slurm_job_id']=='3294' and result['source_commit']==module.SOURCE_COMMIT and
        result['original_failed_terminal_sha256']==module.TERMINAL_SHA256 and
        all(result[name] is False for name in ('accepted','context_verified','production_admission','healthy_checkpoint_accepted',
            'physical_execution_authenticated','health_evaluated','new_evaluation_generated','optimizer_writes_authenticated',
            'model_or_adam_tensors_read','token_journal_read','raw_text_or_ids_or_lp_values_published')),
        'Existing512 identity or non-admission boundary differs')
    raw=module.canonical(result);module.require(len(raw)<=module.MAX_RETURN_BYTES,'Bounded512 result too large')
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    directory=os.open(output.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return dict(output=str(output),sha256=module.sha(raw),remote_executed=True,accepted=False,production_admission=False)


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    for label in ('launch','birth','terminal'):
        parser.add_argument('--'+label,type=Path,required=True);parser.add_argument('--'+label+'-sha256',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    artifacts={label:dict(path=str(getattr(args,label)),sha256=getattr(args,label+'_sha256')) for label in ('launch','birth','terminal')}
    module=helper()
    if args.execute:
        module.require(args.output is not None,'Explicit execute requires exclusive local output')
        result=execute_once(artifacts,args.output,execute=True)
    else:
        reader,_,_=module.input_contract(artifacts);remote_code(module,reader)
        result=dict(role='existing512_local_structure_only',remote_executed=False,accepted=False,production_admission=False)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
