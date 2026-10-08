"""Local-only observation codec tests. Never execute remote code/SSH/NVML."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import snapshot_actual as snapshot


def artifact(raw):
    return dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),physical_path=snapshot.ROOT+'/runs/formal-42/bindings/source.json')


def envelope():
    raw=b'{"source_commit":"188f4e5e7b63cf479290d0888a4a075499886750"}\n'
    resource=b'{"job_id":"42","time":"CPU_FIXTURE"}\n'
    packed=gzip.compress(resource,mtime=0)
    return dict(job_id='42',files={'bindings/source.json':artifact(raw)},
        resource_history=dict(included=True,prefix_bytes=len(resource),
            sha256=hashlib.sha256(resource).hexdigest(),gzip_bytes=len(packed),
            gzip_sha256=hashlib.sha256(packed).hexdigest(),
            gzip_base64=base64.b64encode(packed).decode()))


def test_rendered_remote_body_is_only_read_observation_not_submission():
    source=snapshot.remote_code('42',include_resources=True)
    compile(source,'CPU_RENDER_ONLY','exec')
    assert 'formal-'+ '42' not in snapshot.REMOTE_BODY  # Actual ID is a rendered input, not a frozen guessed job.
    for bad in ('sbatch','scancel','torch','flock','os.exec','os.link','os.kill','--retrieve'):
        assert bad not in source
    assert "--query-compute-apps=gpu_uuid,pid,used_gpu_memory" in source
    assert 'ROOT' not in source.splitlines()[0]  # Literal fixed own root, no shell env expansion.
    assert "'job': '42'" in source.splitlines()[0]


@pytest.mark.parametrize('job',['0','01','-1','42;sbatch','../42','42\n',None,42])
def test_actual_job_selector_rejects_untyped_or_injected_values(job):
    with pytest.raises(ValueError):snapshot.checked_job(job)


def test_original_byte_and_gzip_roundtrip(tmp_path):
    output=tmp_path/'42-CPU_FIXTURE';output.mkdir()
    record=envelope();original=json.dumps(record).encode()
    result=snapshot.save_observation(output,original)
    assert (output/'observed-remote-original.json').read_bytes()==original
    assert (output/'original-files/bindings/source.json').read_bytes()==base64.b64decode(record['files']['bindings/source.json']['base64'])
    assert gzip.decompress((output/'resources-prefix-original.ndjson.gz').read_bytes())==b'{"job_id":"42","time":"CPU_FIXTURE"}\n'
    assert result['transport_original_sha256']==hashlib.sha256(original).hexdigest()
    assert 'base64' not in result['files']['bindings/source.json']
    assert 'gzip_base64' not in result['resource_history']


@pytest.mark.parametrize('kind',['artifact_hash','gzip_hash','gzip_raw_hash','path_escape','wrong_job'])
def test_copied_original_corruption_or_path_escape_rejected(tmp_path,kind):
    output=tmp_path/'42-CPU_FIXTURE';output.mkdir();record=envelope()
    if kind=='artifact_hash':record['files']['bindings/source.json']['sha256']='0'*64
    elif kind=='gzip_hash':record['resource_history']['gzip_sha256']='0'*64
    elif kind=='gzip_raw_hash':record['resource_history']['sha256']='0'*64
    elif kind=='path_escape':record['files']['../escape.json']=record['files'].pop('bindings/source.json')
    elif kind=='wrong_job':record['job_id']='43'
    with pytest.raises(ValueError):snapshot.save_observation(output,json.dumps(record).encode())


def test_duplicate_original_json_refused():
    with pytest.raises(ValueError):snapshot.strict_json(b'{"job_id":"42","job_id":"43"}')


def test_exact_remote_proc_reader_linux_uid_start_job_and_secret_allowlist(tmp_path):
    import ast
    import os
    import re
    tree=ast.parse(snapshot.REMOTE_BODY)
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='process')
    fake=tmp_path/'proc';p=fake/'123';(p/'fd').mkdir(parents=True)
    fields=['S']+['0']*22;fields[19]='1828996485'
    (p/'stat').write_text('123 (python3) '+' '.join(fields)+'\n')
    (p/'status').write_text('Name:\tpython3\nState:\tS (sleeping)\nUid:\t0\t0\t0\t0\n')
    (p/'cgroup').write_text('0::/slurm/uid_0/job_42/step_0\n')
    (p/'environ').write_bytes(b'SLURM_JOB_ID=42\0API_KEY=DO_NOT_COPY\0CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\0')
    (p/'fd/1').symlink_to(tmp_path/'worker-aaa-bbb-123.out')
    namespace={'Path':lambda value: fake if value=='/proc' else Path(value),
               're':re,'os':os,'job':'42','observed':{'processes':{}}}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'EXACT_REMOTE_PROC_CPU_FIXTURE','exec'),namespace)
    actual=namespace['process'](123)
    assert actual['uids']==[0,0,0,0] and actual['start_ticks']=='1828996485'
    assert actual['in_observed_job'] and actual['state']=='S'
    assert actual['environment_allowlist']=={'SLURM_JOB_ID':'42','CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}
    assert 'API_KEY' not in json.dumps(actual) and 'DO_NOT_COPY' not in json.dumps(actual)
    (p/'cgroup').write_text('0::/slurm/uid_0/job_420/step_0\n')
    namespace['observed']['processes']={}
    foreign=namespace['process'](123)
    assert not foreign['in_observed_job'] and 'environment_allowlist' not in foreign
