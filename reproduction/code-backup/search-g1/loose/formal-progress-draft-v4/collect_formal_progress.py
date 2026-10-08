"""External scalar collector. Requires independent exact-byte review first.

The existing sealed submitter observe() is called first with its saved handle.
No execute(), cleanup_source, restart, scheduler mutation, CUDA or training
module imports are reachable from this collector.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import types
import uuid
from pathlib import Path

PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
DRAFT = PREP / 'formal-progress-draft-v4'
INTENT_SHA = 'a9ccca418a9df7922730d96a19bcb239afb13590a8f8418bbce4e9773c040a36'
LAUNCH_SHA = 'ba9b0962cabf2ac322d1386a7e7ad32531770d46f67afe9502f93f00b4088a4e'
SUBMITTER_SHA = '091e9288163ccdf874fa042a87862ac9fd6ce40479d35830ea3c9c43dcf3ed92'
PROTOCOL_SHA = 'b4f80b83ad68b7aae40a8ebeabff963b0f0d5c1080e2d94c428c2ac1e6e4eecf'
FILES = ('collect_formal_progress.py', 'formal_progress_core.py', 'test_formal_progress.py')
LIMIT = 2 * 1024 * 1024

REMOTE_BOOTSTRAP = r'''
import base64,hashlib,json,sys,types
raw=sys.stdin.buffer.read(2*1024*1024+1)
if len(raw)>2*1024*1024:raise ValueError('Scalar RPC input bound exceeded')
def pairs(items):
 value={}
 for key,item in items:
  if key in value:raise ValueError('Duplicate RPC field')
  value[key]=item
 return value
p=json.loads(raw,object_pairs_hook=pairs)
if p['protocol_sha256']!='b4f80b83ad68b7aae40a8ebeabff963b0f0d5c1080e2d94c428c2ac1e6e4eecf':raise ValueError('Protocol SHA differs')
def loaded(name,encoded,expected):
 content=base64.b64decode(encoded,validate=True)
 if len(content)>2*1024*1024 or hashlib.sha256(content).hexdigest()!=expected:raise ValueError('Reviewed code bytes differ')
 module=types.ModuleType(name)
 module.__file__='<reviewed readonly scalar '+name+'>'
 exec(compile(content,module.__file__,'exec'),module.__dict__)
 return module
protocol=loaded('formal_inline_cpu_protocol',p.pop('protocol_b64'),p['protocol_sha256'])
core=loaded('formal_progress_core',p.pop('core_b64'),p['core_sha256'])
result=core.collect_remote(p,protocol)
result['collection_nonce']=p['collection_nonce']
result['collector_core_sha256']=p['core_sha256']
result['reviewed_protocol_sha256']=p['protocol_sha256']
out=core.canonical(result)
if len(out)>2*1024*1024:raise ValueError('Scalar RPC output bound exceeded')
sys.stdout.buffer.write(out+b'\n')
'''


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, item in items:
            need(key not in result, 'Duplicate local proof field')
            result[key] = item
        return result
    def reject(value):
        raise ValueError('Nonfinite proof')
    def number(value):
        result = float(value)
        need(math.isfinite(result), 'Nonfinite proof exponent')
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject, parse_float=number)


def sealed(path, expected):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
         'Canonical ordinary local proof required')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size <= LIMIT,
         'Local proof file bound exceeded')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
             'Local proof changed before open')
        raw = b''
        while len(raw) <= LIMIT:
            block = os.read(fd, min(65536, LIMIT - len(raw) + 1))
            if not block:
                break
            raw += block
        after = path.lstat()
        need(len(raw) == before.st_size and
             (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
             'Local proof changed during read')
    finally:
        os.close(fd)
    need(re.fullmatch('[0-9a-f]{64}', expected or '') is not None and sha(raw) == expected,
         'Exact reviewed local bytes changed: ' + path.name)
    return raw


def module_from_bytes(name, path, raw):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def review_files(path, expected):
    review_raw = sealed(path, expected)
    review = strict_json(review_raw)
    need(type(review) is dict and review.get('role') == 'formal_progress_independent_static_review' and
         review.get('approved') is True and review.get('scope') == 'staticonly' and
         review.get('blocking_findings') == [] and
         review.get('draft_code_executed') is False,
         'Independent static review without draft execution required')
    mapping = review.get('external_file_sha256')
    need(type(mapping) is dict and set(mapping) == set(FILES),
         'Review must bind all three exact collector/test files')
    contents = {name: sealed(DRAFT / name, mapping[name]) for name in FILES}
    need(Path(__file__).resolve() == DRAFT / FILES[0], 'Collector must run from its reviewed draft path')
    return review, contents


def save_unique(receipt):
    path = DRAFT / ('formal-progress-observation-' + str(time.time_ns()) + '-' + uuid.uuid4().hex + '.json')
    raw = canonical(receipt) + b'\n'
    with path.open('xb') as stream:
        need(stream.write(raw) == len(raw), 'Short local scalar receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(DRAFT, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return path, sha(raw)


def collect(review_path, review_sha):
    review, contents = review_files(review_path, review_sha)
    intent_raw = sealed(PREP / 'formal-inline-native-submit-intent.json', INTENT_SHA)
    launch_raw = sealed(PREP / 'formal-inline-native-launch.json', LAUNCH_SHA)
    intent, launch = strict_json(intent_raw), strict_json(launch_raw)
    core = module_from_bytes('formal_progress_core', DRAFT / 'formal_progress_core.py', contents['formal_progress_core.py'])
    need(launch['base'] == intent['base'] == core.BASE and launch['run_id'] == core.RUN and
         launch['job_id'] == core.JOB and launch['source_commit'] == intent['source_commit'] == core.COMMIT and
         launch['source_sha256'] == intent['source_sha256'] and
         launch['source_map_sha256'] == intent['source_map_sha256'] == core.MAP_SHA and
         launch['source_count'] == intent['source_count'] == 372 and
         intent['external_orchestrator_sha256'] == SUBMITTER_SHA and
         intent['cpu_producer_sha256'] == PROTOCOL_SHA,
         'Only exact saved native3309 formal intent and launch are supported')
    submitter_raw = sealed(PREP / 'submit_formal_inline_native.py', SUBMITTER_SHA)
    protocol_raw = sealed(PREP / 'test_formal_inline_target_cpu.py', PROTOCOL_SHA)
    submitter = module_from_bytes('reviewed_formal_inline_submitter', PREP / 'submit_formal_inline_native.py', submitter_raw)
    # Mandatory first live operation; cleanup is explicitly false. The saved
    # handle already exists, and no arbitrary job/base/controller is accepted.
    try:
        observed = submitter.observe(prep=PREP, cleanup_source=False)
    except Exception as error:
        failure = dict(schema_version=1, role='external_readonly_formal_progress_receipt',
            run_id=core.RUN, job_id=core.JOB, base=core.BASE,
            intent_sha256=INTENT_SHA, launch_sha256=LAUNCH_SHA,
            external_file_sha256=review['external_file_sha256'],
            independent_review_sha256=review_sha, collection_state='unknown_or_invalid',
            observation_incomplete=True, native_observer_error_type=type(error).__name__,
            native=dict(state='unknown', exit_code=None, terminal_success='unknown'),
            scalars=None, hourly_anchor_advanced=False, actual_total_writes_unavailable=True,
            scope='mandatory existing observe failed; saved handle retained and no scalar RPC attempted')
        path, digest = save_unique(failure)
        return dict(receipt_path=str(path), receipt_sha256=digest,
                    collection_state=failure['collection_state'], observation_incomplete=True)
    need(sealed(PREP / 'formal-inline-native-submit-intent.json', INTENT_SHA) == intent_raw and
         sealed(PREP / 'formal-inline-native-launch.json', LAUNCH_SHA) == launch_raw,
         'Saved exact handle changed while observing')
    need(observed.get('submission_outcome_known') is True and
         observed.get('source_hashes_match') is True and observed.get('job_id') == core.JOB and
         observed.get('base') == core.BASE and submitter.actual_launch(intent, observed) == launch,
         'Existing observation did not authenticate this exact saved run')
    fields = submitter.job_fields(observed['job_stdout']) if observed['job_returncode'] == 0 else None
    native = core.native_summary(observed, fields)
    births, namespace_provenance = core.namespace_birth_component(
        observed.get('actual_active_namespace_processes'))
    nonce = uuid.uuid4().hex
    payload = dict(base=core.BASE, run_id=core.RUN, job_id=core.JOB,
        source_commit=core.COMMIT, source_sha256=intent['source_sha256'],
        source_map_sha256=core.MAP_SHA, method_sha256=intent['method_sha256'],
        collection_nonce=nonce, observed_namespace_births=births,
        namespace_births_provenance=namespace_provenance,
        protocol_sha256=PROTOCOL_SHA, protocol_b64=base64.b64encode(protocol_raw).decode(),
        core_sha256=sha(contents['formal_progress_core.py']),
        core_b64=base64.b64encode(contents['formal_progress_core.py']).decode())
    receipt = dict(schema_version=1, role='external_readonly_formal_progress_receipt',
        run_id=core.RUN, job_id=core.JOB, base=core.BASE,
        intent_sha256=INTENT_SHA, launch_sha256=LAUNCH_SHA,
        submitter_sha256=SUBMITTER_SHA, reviewed_protocol_sha256=PROTOCOL_SHA,
        external_file_sha256=review['external_file_sha256'],
        independent_review_sha256=review_sha,
        native_observation_document_sha256=sha(submitter.canonical(observed) + b'\n'),
        native_observation_checked_unix=observed['checked_unix'],
        native_observation_finished_unix=observed['observation_finished_unix'],
        native_namespace_births_provenance=namespace_provenance,
        native=native, remote_code_sha256=sha(REMOTE_BOOTSTRAP.encode()),
        collection_nonce=nonce, hourly_anchor_advanced=False,
        terminal_success='not_inferred', actual_total_writes_unavailable=True)
    try:
        result = submitter.remote('h100-8', REMOTE_BOOTSTRAP, payload,
                                  timeout=90, producer_sha256=PROTOCOL_SHA)
        need(result.get('collection_nonce') == nonce and result.get('run_id') == core.RUN and
             result.get('slurm_job_id') == core.JOB and result.get('source_map_sha256') == core.MAP_SHA and
             result.get('runtime_source_sha256') == intent['source_sha256'] and
             result.get('collector_core_sha256') == payload['core_sha256'] and
             result.get('reviewed_protocol_sha256') == PROTOCOL_SHA,
             'Returned scalar provenance differs')
        completeness = dict(result['capture_completeness'])
        components = dict(completeness['components'])
        native_complete = native.get('state') != 'unknown'
        components['native_record'] = dict(status='complete' if native_complete else 'incomplete',
                                            capture_complete=native_complete)
        completeness.update(components=components,
            overall_capture_complete=all(row['capture_complete'] for row in components.values()))
        completeness['observation_incomplete'] = not completeness['overall_capture_complete']
        receipt.update(scalars=result, collection_state='observed',
            capture_completeness=completeness,
            observation_incomplete=completeness['observation_incomplete'])
    except Exception as error:
        receipt.update(collection_state='unknown_or_invalid', observation_incomplete=True,
            scalar_error_type=type(error).__name__, scalars=None,
            scope='saved native observation retained; failed scalar read grants no zero, restart, or success inference')
    path, digest = save_unique(receipt)
    return dict(receipt_path=str(path), receipt_sha256=digest,
                collection_state=receipt['collection_state'],
                observation_incomplete=receipt['observation_incomplete'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true', required=True)
    parser.add_argument('--review-proof', type=Path, required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.review_proof, args.review_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
