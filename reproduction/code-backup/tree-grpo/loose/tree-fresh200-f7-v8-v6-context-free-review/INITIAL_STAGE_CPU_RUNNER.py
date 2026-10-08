"""Execute exact staging source with tiny metadata/code and local path/host fixtures."""
import base64
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace as N
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'work/tree-fresh200-freeze-f7-v8/stage_code_metadata_cpu.py'
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reviewed_stage_fixture',SOURCE)
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def envelope():
    tarbytes=io.BytesIO()
    with tarfile.open(fileobj=tarbytes,mode='w:gz') as archive:
        raw=b'raise RuntimeError("must never execute staged code")\n'
        info=tarfile.TarInfo('tiny-code.py');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    files={name:(tarbytes.getvalue() if name=='CODE_PACKAGE.tar.gz' else b'{"tiny_cpu_fixture":true}\n') for name in stage.FILES}
    return dict(status='REVIEWED_CODE_METADATA_CPU_STAGE_ONLY',files={name:dict(bytes=len(raw),sha256=sha(raw),base64=base64.b64encode(raw).decode()) for name,raw in files.items()})

cases=[]
def call_case(name,mutate=None,host='h100-9',existing=False,symlink_parent=False,wrong_envelope_sha=False,fsync_failure=False):
    value=envelope()
    if mutate:mutate(value)
    raw=json.dumps(value,sort_keys=True).encode();provided='0'*64 if wrong_envelope_sha else sha(raw)
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory);parent=base/'instruct300'
        if symlink_parent:
            real=base/'real';real.mkdir();parent.symlink_to(real,target_is_directory=True)
        else:parent.mkdir()
        dest=parent/'owned-stage'
        if existing:dest.mkdir();(dest/'sentinel').write_bytes(b'preserve')
        error=None;returned=None;captured=io.StringIO();umask=os.umask(0o077)
        try:
            with patch.object(stage,'DESTINATION',dest),patch.object(stage.socket,'gethostname',lambda:host),patch.object(stage.sys,'argv',['stage.py',provided]),patch.object(stage.sys,'stdin',N(buffer=io.BytesIO(raw))),redirect_stdout(captured):
                if fsync_failure:
                    with patch.object(stage.os,'fsync',side_effect=OSError('synthetic fsync failure')):stage.main()
                else:stage.main()
            returned=json.loads(captured.getvalue())
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        finally:os.umask(umask)
        row=dict(name=name,error=error,returned_status=None if returned is None else returned['status'],files_created=sorted(p.name for p in dest.iterdir()) if dest.is_dir() else [],actual_remote_executed=False,actual_gpu_executed=False)
        if name=='valid_exact_inventory':
            assert error is None and returned['status']=='CODE_METADATA_STAGED_ONLY_NOT_EXECUTED'
            assert len(returned['files'])==10 and set(returned['files'])==stage.FILES
            for filename,record in value['files'].items():
                saved=(dest/filename).read_bytes();assert sha(saved)==record['sha256'] and len(saved)==record['bytes']
            assert (dest.stat().st_mode & 0o777)==0o700
            assert all((p.stat().st_mode & 0o777)==0o600 for p in dest.iterdir())
            assert returned['asset_transfer'] is False and returned['slurm_submitted'] is False and returned['optimizer_writes']==0
            row.update(all10_bytes_verified=True,destination_mode='0700',files_mode='0600',receipt=returned,staged_tar_not_extracted_or_executed=True)
        else:
            assert error is not None
            if existing:assert (dest/'sentinel').read_bytes()==b'preserve' and row['files_created']==['sentinel']
            if fsync_failure:assert 'STAGING_RECEIPT.json' not in row['files_created']
            else:assert not row['files_created'] or existing
        cases.append(row)

call_case('valid_exact_inventory')
call_case('wrong_host',host='other-node')
call_case('wrong_envelope_sha',wrong_envelope_sha=True)
call_case('extra_inventory',mutate=lambda value:value['files'].update({'../unsafe.py':value['files']['fresh200_dispatch_common.py']}))
call_case('missing_inventory',mutate=lambda value:value['files'].pop('SOURCE_BYTE_AUDIT.json'))
call_case('wrong_member_sha',mutate=lambda value:value['files']['SOURCE_BYTE_AUDIT.json'].update(sha256='0'*64))
call_case('wrong_member_size',mutate=lambda value:value['files']['SOURCE_BYTE_AUDIT.json'].update(bytes=999))
call_case('invalid_base64',mutate=lambda value:value['files']['SOURCE_BYTE_AUDIT.json'].update(base64='?'))
call_case('existing_destination',existing=True)
call_case('symlink_parent',symlink_parent=True)
call_case('fsync_failure_retains_no_success_receipt',fsync_failure=True)
report=dict(status='STAGE_SOURCE_TINY_CPU_REVIEW_PASSED',stage_source=dict(path=str(SOURCE),sha256=sha(SOURCE.read_bytes())),cases=cases,case_count=len(cases),scope='Exact source main with only hostname/destination/stdin/argv CPU fixtures; 10 tiny code/metadata files. No official model/data or remote command.',actual_package_or_envelope_reviewed=False,actual_remote_executed=False,actual_gpu_executed=False,actual_slurm_executed=False,actual_transfer_allowed=False,limits=['Envelope member names do not semantically validate compressed archive contents; future exact envelope/archive review supplies that binding.','No fsync of directory after staging; file fsync/source no-clobber tested, not host power-loss durability.'])
(OUT/'STAGE_SOURCE_CPU_PROOF.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status=report['status'],case_count=len(cases),actual_remote_executed=False,actual_transfer_allowed=False)))
