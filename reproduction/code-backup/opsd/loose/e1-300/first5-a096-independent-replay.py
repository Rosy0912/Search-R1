from pathlib import Path
import hashlib,importlib.util,json,sys,subprocess
W=Path(__file__).resolve().parent;D=W/'rebound-a0968c2-candidate/first5';OLD='89beac273f783d8e48d7d3e34da040f753514d90';NEW='a0968c2d24beeaa0400f60e4e34677fe6deef9c9';H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();P=D/'manifest.json';assert H(P)=='b4d0cc217253bccaf3aa53c467ed6103ee2ee9375935b84c81a96d9814dede66';m=json.loads(P.read_text())
for n,h in m['tools'].items():
 assert H(D/n)==h
 old=(W/'first5-tools-service-marker-live-state-v2'/n).read_bytes();assert old.count(OLD.encode())==1;assert old.replace(OLD.encode(),NEW.encode())==(D/n).read_bytes()
 spec=importlib.util.spec_from_file_location(Path(n).stem,D/n);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);assert module.SOURCE==NEW
repo=W.parent/'opsd-e1-300';assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==NEW
changes=[]
for n,h in m['source_files'].items():
 assert H(repo/n)==h
 old=subprocess.check_output(['git','-C',str(repo),'show',OLD+':'+n]);
 if hashlib.sha256(old).hexdigest()!=h:changes.append(n)
assert changes==['search_r1/baselines/actor.py','search_r1/baselines/checkpoint_gate.py']
assert H(W/'first5-payload-tools-v3/candidate-manifest.json')==m['native_payload_tools_manifest_sha256']
assert H(W/'first5-payload-tools-v3/inspect_payloads.py')==m['native_payload_inspector_sha256']
print(json.dumps({'scope':'CPU_ONLY_SOURCE_REBIND','manifest_sha256':H(P),'loaded_runtime_modules':{n:sys.modules[n].__file__ for n in ['receipt_payload','hf_cpu_reload']},'changed_source_files':changes,'runtime_delta':'exactly one SOURCE literal per file'},sort_keys=True),flush=True)
import pytest
rc=pytest.main([str(W/'first5-tools-service-marker-live-state-v2/test_first5_tools.py'),'-q'])
for n,h in m['tools'].items():assert H(D/n)==h
assert H(P)=='b4d0cc217253bccaf3aa53c467ed6103ee2ee9375935b84c81a96d9814dede66'
sys.exit(rc)
