"""Remaining prepare/tokenizer route CPU mocks only; call intercepted, Popen blocked."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PACK=ROOT/'work/tree-fresh200-freeze-02b1-v10'
def sha(raw):return hashlib.sha256(raw).hexdigest()
packet_raw=(PACK/'CPU_EXECUTION_PACKET.json').read_bytes();route_raw=(PACK/'CPU_MANAGEMENT_ROUTE_ARGV.json').read_bytes()
assert sha(packet_raw)=='a21179038cd5ce1820092d783ccd5a0ef03eb77b099e38d09cd3354502f32f35'
assert sha(route_raw)=='87acc22d496a8ef4202e7039fc6af5b351d31f2a1f288d8a1d2b0f6bcc21072c'
packet=json.loads(packet_raw);route=json.loads(route_raw);remote=Path(route['route_source'])
source=remote.read_bytes();assert sha(source)==route['route_source_sha256']=='6f715639275c826afc1736863a12a2da0268b98617f174f3d29efa18b682b30b'
tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.If))
scope={'__name__':'independent_mock_only','__file__':str(remote)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[n for n in tree.body if n is not main],type_ignores=[])),str(remote),'exec'),scope)
actual_route=scope['route'];events=[];started=time.process_time()
def prohibited_process(*args,**kwargs):raise RuntimeError('Process launch escaped the call mock; fail closed')
for role,key in [('prepare','prepare'),('tokenizer','tokenizer_preflight')]:
    local=route[role+'_local_argv'];command=packet[key+'_remote_command']
    assert local==['python3','-B',str(remote),command] and '--node' not in local
    assert shlex.split(command)==packet[key+'_argv'] and command==shlex.join(packet[key+'_argv'])
    node_events=[]
    def route_capture(actual_command,node):
        assert actual_command==command and node=='h100-9';node_events.append(node)
        return actual_route(actual_command,node)
    scope['route']=route_capture
    def mocked_call(argv,**kwargs):
        assert argv==actual_route(command,'h100-9')
        stdin=kwargs.get('stdin');data=stdin.read() if stdin is not None else None
        assert data in (None,b'')
        events.append(dict(role=role,node='h100-9',command_sha256=sha(command.encode()),
            stdin_empty=True,subprocess_call_mocked=True,Popen_blocked=True,actual_network=False))
        return 0
    with patch.object(subprocess,'call',mocked_call),patch.object(subprocess,'run',prohibited_process),patch.object(subprocess,'Popen',prohibited_process),patch.object(sys,'argv',[str(remote),command]),contextlib.redirect_stdout(io.StringIO()):
        scope['__name__']='__main__'
        try:exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(remote),'exec'),scope)
        except SystemExit as result:assert result.code in (None,0)
    assert node_events==['h100-9']
assert len(events)==2
report=dict(status='REMAINING_PREPARE_AND_TOKENIZER_ROUTES_CPU_MOCK_PASSED',events=events,
    packet_sha256=sha(packet_raw),route_argv_sha256=sha(route_raw),route_source_sha256=sha(source),
    no_stage_retry=True,actual_remote_executed_in_this_corrected_probe=False,
    actual_slurm_executed=False,actual_gpu_executed=False,actual_assets_loaded=False,
    actual_ready=False,training_release=False,credentials_not_printed=True,
    process_cpu_seconds=time.process_time()-started)
(OUT/'REMAINING_ROUTE_MOCK_RESULT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status='PASSED',mock_calls=2,actual_remote=False,process_cpu_seconds=report['process_cpu_seconds'])))
