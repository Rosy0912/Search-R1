#!/usr/bin/env python3
"""Own Linux CPU contracts incl actual two-process peer; no GPU/algorithm approval."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from protocol import file_sha,require,write_private_json
ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')


def main():
    require(HERE==ROOT/'tools/topd-sharded-ipc','Foreign external tool directory')
    path=HERE.parent/'topd-sharded-scorer/native-cpu-gate.py'
    spec=importlib.util.spec_from_file_location('frozen_scorer_cpu_preflight',path)
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    _,dependencies,versions,paths=frozen.native_preflight()
    package=json.loads((HERE/'files.json').read_text())
    require(package['scope']=='CPU_ONLY_SHARED_IPC_NOT_GPU_READY' and package['expected_native_cpu_fixtures']==48,'Wrong package scope/count')
    require(set(package['files'])=={'protocol.py','session.py','client.py','server.py','test_ipc.py','linux-peer-fixture.py','native-cpu-gate.py','native-cpu.sbatch','README.md'},'Incomplete prototype package')
    for name,sha in package['files'].items():
        require(file_sha(HERE/name)==sha,'Frozen prototype bytes differ: '+name)
        print(sha+'  '+str(HERE/name),flush=True)
    import pytest,torch
    require(not torch.cuda.is_initialized(),'Parent CUDA context forbidden')
    job=os.environ['SLURM_JOB_ID'];output=ROOT/'logs'/('topd-sharded-ipc-native-cpu-'+job)
    output.mkdir(mode=0o700);os.environ['TOPD_IPC_NATIVE_EVIDENCE_DIR']=str(output)
    plugin=frozen.Collection();started=time.time()
    exitcode=pytest.main([str(HERE/'test_ipc.py'),'-q','-ra'],plugins=[plugin])
    require(exitcode==0 and plugin.collected==plugin.passed==48 and plugin.failed==plugin.skipped==0,'Native IPC48 fixtures failed/skipped or count differs')
    require(not torch.cuda.is_initialized(),'CPU fixture initialized CUDA')
    evidence={str(item.relative_to(output)):file_sha(item) for item in output.rglob('*') if item.is_file()}
    require(any('actual-peer-none/receipt-round2.json'==name for name in evidence),'Missing actual native two-round raw receipts')
    record=dict(scope='CPU_ONLY_SHARED_IPC_NOT_GPU_OR_TRAINING_ACCEPTANCE',status='NATIVE_CPU_CONTRACTS_PASS_NOT_GPU_READY',
        job=job,host=socket.gethostname(),cgroups=paths,versions=versions,code_commit=dependencies['code_commit'],
        source_fingerprint=dependencies['source_fingerprint'],original_method_sha256=dependencies['original_method_sha256'],
        cuda_initialized=False,collected=plugin.collected,passed=plugin.passed,skipped=plugin.skipped,failed=plugin.failed,
        files=package['files'],package_manifest_sha256=file_sha(HERE/'files.json'),actual_peer_evidence=evidence,
        started_epoch=started,completed_epoch=time.time())
    write_private_json(output,'result.json',record)
    print(json.dumps(record,sort_keys=True,allow_nan=False),flush=True)


if __name__=='__main__':main()
