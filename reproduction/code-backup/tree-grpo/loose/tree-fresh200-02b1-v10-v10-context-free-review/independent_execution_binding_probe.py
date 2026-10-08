"""Exact packet metadata + SSH route CPU mocks; no remote command is executed."""
import ast
import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PACK=ROOT/'work/tree-fresh200-freeze-02b1-v10';SOURCE=ROOT/'work/tree-behavior-mu-utf8-v2'
COMMIT='02b1ea2688ff3fb032ee6699003a7e09d651f166'
PINS={'CPU_EXECUTION_PACKET.json':'a21179038cd5ce1820092d783ccd5a0ef03eb77b099e38d09cd3354502f32f35',
 'CPU_MANAGEMENT_ROUTE_ARGV.json':'87acc22d496a8ef4202e7039fc6af5b351d31f2a1f288d8a1d2b0f6bcc21072c',
 'FILE_HASHES_LOCAL_PACKET.json':'19810b2ccaf38fb668dabe737d74f98588f3cc6360cf755bb711fa60c0971c64',
 'CODE_METADATA_ENVELOPE.json':'1c570392d2c53e27d48c7e5cd79ebd751a53c8a5538d0a0a418712c39a998267',
 'PREPARATION_INPUTS.json':'ff464a3d36de01a93ffb92d49b0c2a140dd0d85cc9705820d20183f1f0dba5e7',
 'HISTORICAL_RAW_CPU_PLAN.json':'5ef515b2e8e3422715bd1b27cc14a926d4efbbc62bf9487df27b86678ac21be2',
 'stage_code_metadata_cpu.py':'eed7a38dbd8ea1c31564c85054d13e3d2184734164fcfc525a3864e131f24d6f',
 'CODE_PACKAGE.tar.gz':'e0a87a8f68438aac11b3b59d05e11e45c169b5f825c42a4139b769aeb9f972f6',
 'package-code-only/manifest.json':'e3fee8437d02cca89b4e813b6ce49c3938bf3d98662de1599b5aef476a20fd11'}
def digest(raw):return hashlib.sha256(raw).hexdigest()
def obj(name):return json.loads((PACK/name).read_bytes())
started=time.process_time()
for name,expected in PINS.items():assert digest((PACK/name).read_bytes())==expected,name
packet=obj('CPU_EXECUTION_PACKET.json');route=obj('CPU_MANAGEMENT_ROUTE_ARGV.json');inventory=obj('FILE_HASHES_LOCAL_PACKET.json')
manifest=obj('package-code-only/manifest.json');envelope=obj('CODE_METADATA_ENVELOPE.json');inputs=obj('PREPARATION_INPUTS.json');plan=obj('HISTORICAL_RAW_CPU_PLAN.json')
for name,entry in inventory['files'].items():
    raw=(PACK/name).read_bytes();assert len(raw)==entry['bytes'] and digest(raw)==entry['sha256'],name
assert packet['code_commit']==manifest['code_commit']==inputs['code_commit']==plan['reviewed_source_commit']==COMMIT
for key in ('actual_remote_executed','actual_gpu_executed','actual_slurm_submitted','actual_ready','actual_zero_ready','training_release'):
    assert manifest[key] is False
for key in ('production_thresholds_accepted','actual_3b_training_accepted','long_training_ready','original_sampler_actor_gate_accepted'):
    assert manifest[key] is False
assert manifest['probability_calibration']['environment']=={'TREE_FRESH_INITIALIZATION_ONLY':'1','TREE_PPO_CALIBRATION_ONLY':'1'}
assert manifest['probability_calibration']['source_interface_reviewed'] is True
spec=importlib.util.spec_from_file_location('execution_binding_common',PACK/'fresh200_dispatch_common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.validate_manifest(manifest)
archive=PACK/'CODE_PACKAGE.tar.gz';assert archive.stat().st_size==489132
source_blobs={};member_bytes=0
with tarfile.open(archive,'r:gz') as bundle:
    members=bundle.getmembers();assert len(members)==252 and len({m.name for m in members})==252
    assert {m.name for m in members}==set(manifest['files'])|{'manifest.json'}
    for member in members:
        c.relative_file(member.name);assert member.isfile() and not member.issym() and not member.islnk()
        raw=bundle.extractfile(member).read();member_bytes+=len(raw)
        if member.name=='manifest.json':assert digest(raw)==PINS['package-code-only/manifest.json']
        else:assert digest(raw)==manifest['files'][member.name]
        assert raw==(PACK/'package-code-only'/member.name).read_bytes()
        if member.name.startswith('source/repo/'):source_blobs[member.name[len('source/repo/'):]]=raw
assert len(source_blobs)==236 and len(manifest['wrapper_files'])==15 and member_bytes<=33554432
# One metadata/code batch checks the whole archive source closure against HEAD.
assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD']).decode().strip()==COMMIT
assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain=v1'])
keys=sorted(source_blobs);queries=''.join(COMMIT+':'+name+'\n' for name in keys).encode()
batch=subprocess.check_output(['git','-C',str(SOURCE),'cat-file','--batch'],input=queries);offset=0
for name in keys:
    stop=batch.index(b'\n',offset);header=batch[offset:stop].split();assert header[1]==b'blob'
    size=int(header[2]);offset=stop+1;assert batch[offset:offset+size]==source_blobs[name]
    offset+=size+1
assert offset==len(batch)
assert set(envelope['files'])==set(packet['stage_files'])==set(packet['local_sources']) and len(envelope['files'])==10
for name,record in envelope['files'].items():
    raw=base64.b64decode(record['base64'],validate=True)
    assert digest(raw)==record['sha256']==packet['stage_files'][name]['sha256']
    assert len(raw)==record['bytes']==packet['stage_files'][name]['bytes']
    assert raw==Path(packet['local_sources'][name]).read_bytes()
assert digest((PACK/'SOURCE_CONTEXT_FREE_RECEIPT.json').read_bytes())=='c29fdcb835263b9cf03705cabe5cca80903aba72c20c07e797b8cc70e4e621c5'
assert digest((PACK/'SOURCE_BYTE_AUDIT.json').read_bytes())=='8a21cd2284489d5f90fc44842f9b54169331a384a91ddc94f9f007e3ca61a8e9'
assert digest((PACK/'HELPER_SOURCE_REVIEW.json').read_bytes())=='abf1085dcfa1603830a5d86c2f8e20154d9be4da5f4a6c2a907bd1565e7d39b7'
stage=packet['stage_directory'];target=packet['candidate_target']
assert stage=='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/preparation-02b1-v10-20261006'
assert target=='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-mu-02b1-v10'
assert inputs['archive']==stage+'/CODE_PACKAGE.tar.gz' and inputs['target']==target
assert inputs['archive_sha256']==PINS['CODE_PACKAGE.tar.gz'] and inputs['manifest_sha256']==PINS['package-code-only/manifest.json']
for key,filename in [('source_interface_receipt','SOURCE_INTERFACE_RECEIPT.json'),('context_free_review','SOURCE_CONTEXT_FREE_RECEIPT.json'),('context_byte_audit','SOURCE_BYTE_AUDIT.json')]:
    assert inputs[key]['path']==stage+'/'+filename and inputs[key]['sha256']==digest((PACK/filename).read_bytes())
with tempfile.TemporaryDirectory() as temporary:
    fake=Path(temporary);c.ROOT=fake;mockinputs=copy.deepcopy(inputs)
    for key,filename in [('source_interface_receipt','SOURCE_INTERFACE_RECEIPT.json'),('context_free_review','SOURCE_CONTEXT_FREE_RECEIPT.json'),('context_byte_audit','SOURCE_BYTE_AUDIT.json')]:
        dest=fake/filename;dest.write_bytes((PACK/filename).read_bytes());mockinputs[key]['path']=str(dest)
    assert c.validate_reviews(mockinputs,manifest,for_dispatch=True)['source_review_accepted'] is True
# Stage source is unchanged apart from the explicit new destination.
newstage=ast.parse((PACK/'stage_code_metadata_cpu.py').read_bytes())
oldstage=ast.parse((ROOT/'work/tree-fresh200-freeze-f7-v8/stage_code_metadata_cpu.py').read_bytes())
for tree in (oldstage,newstage):
    statement=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DESTINATION' for t in n.targets))
    statement.value=ast.Constant('EXPLICIT_NEW_REVIEWED_DESTINATION')
assert ast.dump(newstage,include_attributes=False)==ast.dump(oldstage,include_attributes=False)
for name in ('stage','prepare','tokenizer_preflight'):
    command=packet[name+'_remote_command'];argv=packet[name+'_argv']
    assert shlex.split(command)==argv and command==shlex.join(argv)
    assert argv[:7]==['env','CUDA_VISIBLE_DEVICES=','PYTHONOPTIMIZE=0','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1','/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python','-B']
assert packet['stage_argv'][7]=='-c' and packet['stage_argv'][8]==(PACK/'stage_code_metadata_cpu.py').read_text()
assert packet['stage_argv'][9]==PINS['CODE_METADATA_ENVELOPE.json']
assert packet['prepare_argv'][7]==stage+'/prepare_tree_fresh200_package.py'
assert packet['tokenizer_preflight_argv'][7]==target+'/tokenizer_raw_preflight.py'
assert packet['tokenizer_preflight_argv'][-1]==target+'/CPU_PREP_CONFIG/HISTORICAL_RAW_CPU_PREFLIGHT.json'
assert plan['raw_helper']=={'path':target+'/source/repo/search_r1/baselines/tree_raw_tokens.py','sha256':'15b696607846046f671e30531420c6f48cca321762f77016f4e5bdb7868ea229'}
assert [run['job_id'] for run in plan['runs']]==['3295','3296'] and all(len(run['rank_ready'])==8 for run in plan['runs'])
assert len({entry['path'] for run in plan['runs'] for entry in run['rank_ready']})==16
assert route['node']=='h100-9' and route['node_override_argument_present'] is False and route['packet_sha256']==PINS['CPU_EXECUTION_PACKET.json']
remote=Path(route['route_source']);remote_raw=remote.read_bytes()
assert digest(remote_raw)==route['route_source_sha256']=='6f715639275c826afc1736863a12a2da0268b98617f174f3d29efa18b682b30b'
# Never output route source or the constructed route (contains credentials).
tree=ast.parse(remote_raw);main=next(n for n in tree.body if isinstance(n,ast.If))
prefix=ast.Module(body=[n for n in tree.body if n is not main],type_ignores=[])
scope={'__name__':'independent_route_probe','__file__':str(remote)}
exec(compile(ast.fix_missing_locations(prefix),str(remote),'exec'),scope)
actual_route=scope['route'];route_events=[]
def capture_route(command,node):
    assert node=='h100-9';route_events.append(dict(node=node,command_sha256=digest(command.encode())))
    return actual_route(command,node)
scope['route']=capture_route
mock_calls=[]
for role,packet_role in [('stage','stage'),('prepare','prepare'),('tokenizer','tokenizer_preflight')]:
    local=route[role+'_local_argv'];command=packet[packet_role+'_remote_command']
    assert local[:3]==['python3','-B',str(remote)] and '--node' not in local
    assert local[-1]==command
    if role=='stage':assert local[3:5]==['--stdin',str(PACK/'CODE_METADATA_ENVELOPE.json')]
    else:assert len(local)==4
    def mocked_call(argv,**kwargs):
        assert argv==actual_route(command,'h100-9')
        stdin=kwargs.get('stdin');raw=stdin.read() if stdin is not None else None
        if role=='stage':assert type(raw) is bytes and digest(raw)==PINS['CODE_METADATA_ENVELOPE.json']
        else:assert raw is None
        mock_calls.append(dict(role=role,node='h100-9',remote_command_sha256=digest(command.encode()),
            stdin_sha256=digest(raw) if raw is not None else None,stdin_bytes=len(raw) if raw is not None else 0,
            subprocess_mocked=True,credentials_printed=False))
        return 0
    def prohibited_process(*args,**kwargs):
        raise RuntimeError('A process launch escaped the route CPU mock; fail closed')
    with patch.object(subprocess,'call',mocked_call),patch.object(subprocess,'run',prohibited_process),patch.object(subprocess,'Popen',prohibited_process),patch.object(sys,'argv',[str(remote),*local[3:]]),contextlib.redirect_stdout(io.StringIO()):
        scope['__name__']='__main__'
        try:exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(remote),'exec'),scope)
        except SystemExit as exit_result:assert exit_result.code in (None,0)
assert len(route_events)==len(mock_calls)==3
result=dict(status='EXACT_02B1_V10_CPU_EXECUTION_BINDING_PROBE_PASSED',source_commit=COMMIT,
    exact_packet_sha256=PINS['CPU_EXECUTION_PACKET.json'],exact_route_argv_sha256=PINS['CPU_MANAGEMENT_ROUTE_ARGV.json'],
    pins=PINS,inventory_files_verified=len(inventory['files']),envelope_files_verified=10,
    archive_regular_files=252,archive_source_files=236,archive_wrapper_files=15,archive_uncompressed_bytes=member_bytes,
    source_archive_matches_clean_git_head=True,source_receipts_and_manifest_real_api_bind=True,
    quoted_argv_roundtrips_exactly=True,stage_source_only_destination_changed=True,route_mock_calls=mock_calls,
    default_management_node='h100-9',credential_source_never_printed=True,
    historical_expected_rows=5120,historical_records_read_by_this_reviewer=0,
    no_clobber_stage_and_candidate_source_verified=True,remote_destination_existence_actually_checked=False,
    actual_remote_executed=False,actual_slurm_executed=False,actual_gpu_executed=False,actual_assets_loaded=False,
    source_tests_repeated=False,actual_ready=False,actual_zero_ready=False,training_release=False,
    process_cpu_seconds=time.process_time()-started)
(OUT/'CPU_EXECUTION_BINDING_PROBE_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status='PASSED',archive_files=252,source_files=236,envelope_files=10,route_mocks=3,process_cpu_seconds=result['process_cpu_seconds'])))
