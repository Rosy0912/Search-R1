"""Real helper API schema calls; relocates only the mocked same-node root."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def refusal(call):
    try:call()
    except ValueError as error:return str(error)
    raise AssertionError('Expected fail-closed refusal')
started=time.process_time();cases=[]
with tempfile.TemporaryDirectory() as temporary:
    fake=Path(temporary);helpers=fake/'helpers';helpers.mkdir()
    for name in ('fresh200_dispatch_common.py','prepare_tree_fresh200_package.py','submit_tree_fresh200_once.py'):
        (helpers/name).write_bytes((ROOT/'work/tree-fresh200-dispatch-v10'/name).read_bytes())
    helperreview=fake/'HELPER_SOURCE_REVIEW.json';helperreview.write_bytes((OUT/'HELPER_SOURCE_REVIEW.json').read_bytes())
    # Execute unmodified common code, then relocate its environmental ROOT for
    # this CPU mock; this does not modify any file or bypass byte/schema checks.
    def relocated_exec(code,scope):
        builtins.exec(code,scope);scope['ROOT']=fake
    for name in ('prepare_tree_fresh200_package.py','submit_tree_fresh200_once.py'):
        m=module(helpers/name,'compat_'+name.replace('.py',''));m.exec=relocated_exec
        actual=m.load_common(str(helperreview),sha(helperreview));assert actual.COMMIT=='02b1ea2688ff3fb032ee6699003a7e09d651f166'
        bad=json.loads(helperreview.read_text());bad['actual_gpu_accepted']=True
        badpath=fake/'badhelper.json';badpath.write_text(json.dumps(bad));assert 'GPU acceptance' in refusal(lambda:m.load_common(str(badpath),sha(badpath)))
        cases.append(dict(name='real_'+name+'_load_common_exact_three_byte_bindings',passed=True))
    c=module(helpers/'fresh200_dispatch_common.py','compat_common');c.ROOT=fake
    inputs={}
    locations={'source_interface_receipt':ROOT/'work/tree-fresh200-utf8-v2-owner/UTF8_V2_SOURCE_INTERFACE_CPU_RECEIPT.json',
        'context_free_review':OUT/'RECEIPT.json','context_byte_audit':OUT/'BYTE_AUDIT.json'}
    for key,path in locations.items():
        dest=fake/(key+'.json');dest.write_bytes(path.read_bytes());inputs[key]=dict(path=str(dest),sha256=sha(dest))
    audit=json.loads((OUT/'BYTE_AUDIT.json').read_text())
    manifest=dict(files={**{name:entry['sha256'] for name,entry in audit['wrapper_files'].items()},
        **{'source/repo/'+name:entry['sha256'] for name,entry in audit['source_files'].items()}})
    accepted=c.validate_reviews(inputs,manifest,for_dispatch=True);assert accepted['source_review_accepted'] is True
    cases.append(dict(name='real_validate_reviews_accepts_exact_sealed19source15wrapper_receipts',passed=True))
    contextpath=Path(inputs['context_free_review']['path']);original=contextpath.read_bytes()
    changed=json.loads(original);changed['stages']['fresh_mu_calibration']['source_review_accepted']=False
    contextpath.write_text(json.dumps(changed));inputs['context_free_review']['sha256']=sha(contextpath)
    assert c.validate_reviews(inputs,manifest,for_dispatch=False)['source_review_accepted'] is False
    assert 'remains blocked' in refusal(lambda:c.validate_reviews(inputs,manifest,for_dispatch=True))
    cases.append(dict(name='real_validate_reviews_preserves_blocked_source_for_cpu_only',passed=True))
report=dict(status='INDEPENDENT_SEALED_RECEIPTS_HELPER_API_COMPATIBLE_CPU_ONLY',cases=cases,
    receipt_sha256=sha(OUT/'RECEIPT.json'),byte_audit_sha256=sha(OUT/'BYTE_AUDIT.json'),
    helper_source_review_sha256=sha(OUT/'HELPER_SOURCE_REVIEW.json'),process_cpu_seconds=time.process_time()-started,
    actual_remote_executed=False,actual_slurm_executed=False,actual_gpu_executed=False,
    actual_ready=False,actual_assets_loaded=False,environmental_same_node_root_relocated_in_memory_only=True)
(OUT/'RECEIPT_COMPATIBILITY_PROBE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='PASSED',cases=len(cases),process_cpu_seconds=report['process_cpu_seconds'])))
