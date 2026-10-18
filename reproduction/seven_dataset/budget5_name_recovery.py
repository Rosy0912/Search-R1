"""One inspected continuation after the legacy-name validation failure; no retry."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys
import time
import traceback
from .final150_dispatch import digest, remote, parse_submissions, write_new
from .stream_copy import transfer


def validate_prior(prior, proof):
    if (prior.get('pid') != 36239 or prior.get('group') != 'budget-5'
            or prior.get('phase') != 'FAILED_PRESERVE_EVIDENCE_NO_AUTO_RETRY'
            or 'value length must be between 1 and 32' not in prior.get('error', '')
            or any(key in prior for key in ['preflight', 'submissions', 'transfer', 'transfer_started'])):
        raise ValueError('Failure was not the inspected pre-transfer name check')
    for key in ['checkpoint_identity', 'archive_bytes', 'archive_sha256']:
        if prior[key] != proof[key]:
            raise ValueError('Frozen completed export differs: ' + key)


def run(configfile):
    cfg = json.loads(Path(configfile).read_text())
    p = Path(cfg['local'])
    old_cfg = json.loads((p / 'pj-budget5-stream-config-20261018T1640.json').read_text())
    prior = json.loads((p / 'pj-budget5-stream-dispatch.latest.json').read_text())
    proof = json.loads((p / 'pj-budget5-stream-20261018T1640/export.json').read_text())
    validate_prior(prior, proof)
    try:
        os.kill(36239, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Previous controller still exists')
    lock = open(p / 'pj-ablation-dispatch-20261018T1103/lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, expected in cfg['files'].items():
        assert digest(p / name) == expected, name
    run = Path(cfg['run'])
    run.mkdir(exist_ok=False)
    state = dict(at=time.time(), pid=os.getpid(), prior_dispatcher=36239,
        original_dispatcher=13526, source=cfg['source'], worker_source=old_cfg['worker_source'],
        group='budget-5', checkpoint_identity=proof['checkpoint_identity'],
        phase='INDEPENDENT_ZERO_SUBMISSION_CHECK', no_auto_retry=True,
        export_proof=str(p / 'pj-budget5-stream-20261018T1640/export.json'))

    def save():
        state['at'] = time.time()
        path = Path(cfg['state'])
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(path)

    def helper(name):
        assert digest(p / name) == cfg['files'][name]
        return (p / name).read_text()

    save()
    try:
        lumia, pj, commit = old_cfg['lumia'], old_cfg['pj'], old_cfg['worker_source']
        ls, ps, lp, pp, sdk = (old_cfg[k] for k in
            ['lumia_ssh', 'pj_ssh', 'lumia_python', 'pj_python', 'pj_sdk'])
        dest = prior['archive']
        code = ('CID=' + repr(proof['checkpoint_identity']) + '\nB=' + repr(pj)
                + '\nDEST=' + repr(dest) + '\n'
                + helper('source/reproduction/seven_dataset/pj_name_inspection.py')
                + '\n' + helper(cfg['zero_helper']))
        raw = remote(ps, sdk, code, pj + '/sources/' + commit, 180)
        (run / 'zero-submission.output').write_text(raw)
        zero = json.loads(raw.split('ZERO_PROOF=', 1)[1].strip())
        write_new(run / 'zero-submission.json', zero)
        assert len(zero['exact_jobs']) == 6
        state.update(phase='STREAMING_EXISTING_VERIFIED_EXPORT', transfer_started=time.time())
        save()
        alarm = ('import signal,sys,json\ndef expired(*a):raise TimeoutError("Bounded stream; preserve partial")\n'
                 'signal.signal(signal.SIGALRM,expired)\nsignal.alarm(7200)\n')
        send = (alarm + 'from reproduction.seven_dataset.resume_stream import inspect_prefix,send_suffix\n'
                + 'p=' + repr(proof['archive']) + '\ne=inspect_prefix(p,0)\nassert e["size"]=='
                + str(proof['archive_bytes']) + '\nsend_suffix(p,e,sys.stdout.buffer)')
        receive = (alarm + 'from reproduction.seven_dataset.resume_stream import inspect_prefix,receive_suffix\n'
                   + 'p=' + repr(dest) + '\nopen(p,"xb").close()\ne=inspect_prefix(p,0)\n'
                   + 'print(json.dumps(receive_suffix(p,e,' + str(proof['archive_bytes']) + ','
                   + repr(proof['archive_sha256']) + ',sys.stdin.buffer)))')

        def command(ssh, py, code, base):
            return ssh + ['PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='
                + shlex.quote(base + '/sources/' + commit) + ' ' + shlex.quote(py)
                + ' -c ' + shlex.quote(code)]

        state['transfer'] = transfer(command(ls, lp, send, lumia),
            command(ps, pp, receive, pj), 7260)
        state.update(phase='PJ_FROZEN_INPUT_PREFLIGHT', transfer_finished=time.time())
        save()
        raw = remote(ps, pp, 'PROOF=' + repr(proof) + '\nARCHIVE=' + repr(dest)
            + '\nCOMMIT=' + repr(commit) + '\n' + helper(old_cfg['stage_helper']),
            pj + '/sources/' + commit, 1200)
        (run / 'preflight.output').write_text(raw)
        preflight = json.loads(raw.strip().splitlines()[-1])
        write_new(run / 'preflight.json', preflight)
        state.update(phase='SUBMITTING_FIXED_THREE_ONCE', preflight=str(run / 'preflight.json'))
        save()
        raw = remote(ps, sdk, 'PREFLIGHT=' + repr(preflight) + '\n'
            + helper(old_cfg['submit_helper']), pj + '/sources/' + commit, 300)
        (run / 'submission.output').write_text(raw)
        submissions = parse_submissions(raw)
        for item in submissions:
            write_new(p / ('seven-ablation-pj-budget-5-150-r' + str(item['repeat_index'])
                + '-submission.json'), item)
        state.update(phase='THREE_SUBMITTED_NO_REPLAY', submissions=submissions)
        save()
    except Exception as exc:
        state.update(phase='FAILED_PRESERVE_EVIDENCE_NO_AUTO_RETRY', error=repr(exc),
                     traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    run(sys.argv[1])
