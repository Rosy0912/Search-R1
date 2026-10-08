"""User-authorized transient SSH transport for the unchanged reviewed V6 launcher.
Latest user explicitly requested direct execution without further reviews/tests.
No config, known_hosts, training-source, gate, method or review-proof edits.
The V6 static reviews do not claim to review this transport override.
"""
import hashlib, importlib.util, json, shlex, sys
from pathlib import Path
ROOT=Path('/Users/shaoshitong/Documents/Search-G1/preparation/numeric-repair-native-v6')
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control

def route(host,command):
    if host not in ('h100-8','h100-9'):
        raise ValueError('Only existing target/source hosts are allowed')
    inner=shlex.join(['env','-u','LD_LIBRARY_PATH','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',host,command])
    head=shlex.join(['env','-u','LD_LIBRARY_PATH','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','h100-16',inner])
    return control.ssh('mgmt-host')+[head]

if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('--execute','--observe','--cleanup-source'):
        raise ValueError('Use one saved-attempt mode')
    path=ROOT/'submit_formal_inline_native.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='a04126835420059db65e89d7c4ae1d6effb5abf8f451bd9041153a70be995e11':
        raise ValueError('Original reviewed launcher changed')
    mode=sys.argv[1]
    arguments=json.loads((ROOT/'numeric-repair-sealed-args.json').read_text()) if mode=='--execute' else []
    control.route=route
    spec=importlib.util.spec_from_file_location('unchanged_formal_native_v6',path)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    sys.argv=[str(path),mode]+arguments
    native.main()
