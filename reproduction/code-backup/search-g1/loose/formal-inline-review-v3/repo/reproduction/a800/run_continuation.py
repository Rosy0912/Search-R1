"""Exclusive launch of the reviewed checkpoint200 -> logical300 new segment."""
import argparse
import fcntl
import json
import os
import signal
import socket
from pathlib import Path
from reproduction.a800 import run_pipeline
from reproduction.a800.prepare_continuation import prepare_segment,lineage_lock_root
from reproduction.a800.prepare_run import digest
from reproduction.a800.owned_processes import publish
from reproduction.a800.recovery_binding import intervention_fingerprint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-commit',required=True);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--dumps',type=Path,required=True);ap.add_argument('--completed',type=int,choices=(200,250),default=200);ap.add_argument('--calibration-acceptance',type=Path)
    args=ap.parse_args();base=Path(os.environ['SEARCH_G1_BASE'])
    assert base.resolve()!=args.parent.resolve()
    for name in ('logs','tmp','ray','cache/triton','cache/inductor','cache/hf','artifacts'):(base/name).mkdir(parents=True,exist_ok=True)
    # Share the ORIGINAL physical-device lock; private new-base lock alone would
    # allow an original and continuation supervisor to contend for the same GPUs.
    parent_run=json.loads((args.parent/'logs/run-manifest.json').read_text())
    lock_root=lineage_lock_root(args.parent)
    lock=(lock_root/'gpu-job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (base/'logs/process-ledger.json').exists(), 'Existing segment ledger: inspect/cleanup, never duplicate launch'
    assert not list((base/'artifacts').glob('round_*'))
    assert run_pipeline.r1_finished()
    run_pipeline.assert_idle()
    for port in (18010,18012):
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    if args.completed==200:
        assert args.dumps.resolve().is_relative_to((args.parent/'artifacts').resolve())
    else:
        from reproduction.a800.recovery_binding import verify_repair_calibration
        assert args.calibration_acceptance is not None
        verify_repair_calibration(args.dumps,args.calibration_acceptance)
    proof=json.loads((args.dumps/'calibration-label-proof.json').read_text())
    assert Path(proof['model']).resolve()==(Path(parent_run['checkpoint_dir'])/'actor'/f'global_step_{args.completed}').resolve()
    run=prepare_segment(args.parent,args.source_commit,args.completed)
    assert run['initial_probe_version'] is None and not Path(run['train_log']).exists()
    run['recovery_dumps']=str(args.dumps.resolve())
    if args.completed==250:
        run['calibration_acceptance']=str(args.calibration_acceptance.resolve())
        run['calibration_acceptance_sha256']=digest(args.calibration_acceptance)
    run['recovery_dump_sha256']=intervention_fingerprint(args.dumps)
    publish(base/'logs/run-manifest.json',run)
    signal.signal(signal.SIGTERM,run_pipeline.request_interrupt)
    signal.signal(signal.SIGINT,run_pipeline.request_interrupt)
    run_pipeline.Pipeline(base,run).execute()


if __name__=='__main__':main()
