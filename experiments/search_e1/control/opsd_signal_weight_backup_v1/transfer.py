"""HKUST4→HKUST1 SSH pipe; checkpoint bytes never become a Mac file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

try:from . import peer
except ImportError:import peer


class TransferError(RuntimeError):
    def __init__(self,message,evidence):
        super().__init__(message);self.evidence=evidence


def ssh_command(alias,python,script,peer_sha,action,args):
    peer.require(alias in ('hkust4','hkust1') and Path(python).is_absolute() and Path(script).is_absolute(),
        'Only explicit approved HKUST peers and absolute published Python/script paths')
    peer.require(len(peer_sha)==64 and all(c in '0123456789abcdef' for c in peer_sha),'Explicit published peer source SHA required')
    remote=['env','CUDA_VISIBLE_DEVICES=','E1_RUN_ID=signal-backup-independent','PYTHONDONTWRITEBYTECODE=1',python,'-B',script,'--peer-sha256',peer_sha,action]+list(args)
    return ['ssh']+list(peer.DIRECT_SSH_OPTIONS)+[alias,shlex.join(remote)]


class DiagnosticDrain:
    """Continuously drain source diagnostics so tar cannot deadlock behind stderr.

    The tar/stdout payload remains an unread local pipe. Capture bounds apply to
    diagnostic bytes only; even overflow is fully drained and SHA-recorded.
    """
    def __init__(self,stream,cap=128<<10):
        self.stream=stream;self.cap=cap;self.captured=bytearray();self.total=0;self.digest=hashlib.sha256();self.error=None
        self.thread=threading.Thread(target=self.consume,name='hkust-backup-diagnostic-drain',daemon=False);self.thread.start()

    def consume(self):
        try:
            while True:
                block=self.stream.read(8192)
                if not block:break
                self.total+=len(block);self.digest.update(block)
                if len(self.captured)<self.cap:self.captured.extend(block[:self.cap-len(self.captured)])
        except Exception as error:self.error=type(error).__name__

    def finish(self):
        self.thread.join()
        return bytes(self.captured),dict(bytes=self.total,sha256=self.digest.hexdigest(),capture_complete=self.total<=self.cap,error_type=self.error)


def read_call(command,runner=subprocess.run):
    result=runner(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    peer.require(len(result.stdout)<=128<<10 and len(result.stderr)<=128<<10,'Remote small control output exceeds128KiB')
    evidence=dict(actual_ssh_wait_returncode=result.returncode,
        stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
    if result.returncode!=0:raise TransferError('Actual remote control SSH returned nonzero; preserve pending evidence',evidence)
    return json.loads(result.stdout),evidence


def source_terminal(stderr):
    matches=[]
    for line in stderr.splitlines():
        try:value=json.loads(line)
        except (ValueError,UnicodeError):continue
        if 'source_stream_terminal' in value:matches.append(value['source_stream_terminal'])
    peer.require(len(matches)==1,'Exactly one actual published source stream terminal is required')
    return matches[0]


def backup(config,config_sha,checkpoint,checkpoint_sha,variant,updates,outer,source_python,source_script,target_python,target_script,peer_sha,
           runner=subprocess.run,popen=subprocess.Popen):
    peer.require('torch' not in sys.modules,'Transport controller must not import Torch')
    start=time.time();source=lambda action,args:ssh_command('hkust4',source_python,source_script,peer_sha,action,args)
    target=lambda action,args:ssh_command('hkust1',target_python,target_script,peer_sha,action,args)
    inventory_args=['--config',config,'--config-sha256',config_sha,'--checkpoint',checkpoint,'--checkpoint-sha256',checkpoint_sha,
        '--variant',variant,'--expected-updates',str(updates),'--expected-outer',str(outer)]
    value,inventory_wait=read_call(source('inventory',inventory_args),runner)
    peer.require(value['accepted'] is True and value['variant']==variant and value['optimizer_updates']==updates
        and value['outer_index']==outer and value['config']['path']==config and value['config']['sha256']==config_sha
        and value['checkpoint']['path']==checkpoint and value['checkpoint']['sha256']==checkpoint_sha
        and value['peer_source_sha256']==peer_sha,'Actual remote pinned inventory differs from explicit request')
    encoded=peer.encode(value);probe,probe_wait=read_call(target('probe',['--inventory-base64',encoded]),runner)
    if probe.get('accepted') is True:
        return dict(accepted=True,already_complete=True,remote_backup=probe,inventory_sha256=peer.object_sha(value),
            inventory_actual_wait=inventory_wait,target_probe_actual_wait=probe_wait,local_checkpoint_payload_files_created=0,
            no_gpu_or_training_started=True,no_signals_sent=True,elapsed_seconds=time.time()-start)
    peer.require(probe['state']=='NOT_COMPLETE','Target has unresolved or inconsistent prior accepted evidence')
    # The local descriptor joins the two remote processes. No local checkpoint
    # path, tar file, model deserialization, tempfile or payload stdout capture.
    sender=popen(source('send',['--inventory-base64',encoded]),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    drain=DiagnosticDrain(sender.stderr)
    receiver=None
    try:
        receiver=popen(target('receive',['--inventory-base64',encoded]),stdin=sender.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    except Exception as error:
        sender.stdout.close()  # No receiver exists: normal broken pipe ends this owned source stream.
        source_code=sender.wait();_,diagnostics=drain.finish()
        raise TransferError('Target SSH could not launch; actual owned sender waited and remote partial left intact',
            dict(source_ssh_actual_wait_returncode=source_code,target_ssh_actual_wait_returncode='NOT_LAUNCHED',
                target_launch_error_type=type(error).__name__,source_stderr=diagnostics,local_checkpoint_payload_files_created=0)) from error
    else:
        sender.stdout.close()  # Receiver owns its inherited pipe; parent never reads/checkpoints these bytes.
    target_stdout_drain=DiagnosticDrain(receiver.stdout);target_stderr_drain=DiagnosticDrain(receiver.stderr)
    target_code=receiver.wait();receive_stdout,target_stdout_diag=target_stdout_drain.finish()
    receive_stderr,target_stderr_diag=target_stderr_drain.finish()
    source_code=sender.wait();send_stderr,diagnostics=drain.finish()
    transport=dict(source_alias='hkust4',target_alias='hkust1',source_ssh_actual_wait_returncode=source_code,
        direct_ssh_options=list(peer.DIRECT_SSH_OPTIONS),connection_mode='direct_no_proxy_no_multiplex',
        target_ssh_actual_wait_returncode=target_code,source_stderr_sha256=diagnostics['sha256'],source_stderr=diagnostics,
        target_stdout_sha256=target_stdout_diag['sha256'],target_stderr_sha256=target_stderr_diag['sha256'],
        target_stdout=target_stdout_diag,target_stderr=target_stderr_diag,
        local_checkpoint_payload_files_created=0,large_output_target=peer.TARGET,checked_unix=time.time())
    if source_code!=0 or target_code!=0:
        raise TransferError('Actual tar transfer SSH nonzero; remote attempt/partial is preserved, no completion marker requested',transport)
    if any(not d['capture_complete'] or d['error_type'] for d in (diagnostics,target_stdout_diag,target_stderr_diag)):
        raise TransferError('Control diagnostics overflow/reader error preserved; transfer not accepted',transport)
    terminal=source_terminal(send_stderr);pending=json.loads(receive_stdout)
    peer.require(pending['accepted'] is True and pending['complete'] is False and terminal['accepted'] is True
        and terminal['source_tar_actual_wait_returncode']==0 and terminal['before_after_inventory_equal'] is True
        and terminal['inventory_sha256']==peer.object_sha(value),'Remote pending/source actual terminal does not establish byte-stable wait0 transfer')
    ref=pending['pending_receipt']
    result,commit_wait=read_call(target('commit',['--inventory-base64',encoded,'--pending-receipt',ref['path'],
        '--pending-receipt-sha256',ref['sha256'],'--source-terminal-base64',peer.encode(terminal),'--transport-base64',peer.encode(transport)]),runner)
    peer.require(result['accepted'] is True and result['state']=='COMPLETE_FRESH_SHA_READBACK','Target did not accept full fresh-SHA member coverage')
    return dict(accepted=True,already_complete=False,remote_backup=result,inventory_sha256=peer.object_sha(value),
        inventory_actual_wait=inventory_wait,target_probe_actual_wait=probe_wait,actual_source_stream=terminal,actual_transport=transport,
        target_commit_actual_wait=commit_wait,local_checkpoint_payload_files_created=0,no_gpu_or_training_started=True,no_signals_sent=True,
        elapsed_seconds=time.time()-start,
        scope='Pinned immutable checkpoint byte backup only; source/debug assets remain preserved, no optimizer numeric or standalone resume acceptance')


def main():
    p=argparse.ArgumentParser()
    for name in ('config','config-sha256','checkpoint','checkpoint-sha256','variant','source-python','source-script','target-python','target-script','peer-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--expected-updates',required=True,type=int);p.add_argument('--expected-outer',required=True,type=int)
    p.add_argument('--audit-output');a=p.parse_args()
    try:
        result=backup(a.config,a.config_sha256,a.checkpoint,a.checkpoint_sha256,a.variant,a.expected_updates,a.expected_outer,
            a.source_python,a.source_script,a.target_python,a.target_script,a.peer_sha256)
    except Exception as error:
        result=dict(accepted=False,error_type=type(error).__name__,error=str(error),local_checkpoint_payload_files_created=0,
            remote_partial_deletion_performed=False,remote_partial_existence='UNKNOWN unless actual receiver evidence was returned',
            no_signals_sent=True,no_gpu_or_training_started=True)
        if isinstance(error,TransferError):result['failed_actual_transport_evidence']=error.evidence
    if a.audit_output:result['local_small_audit']=peer.write_once(Path(a.audit_output),result)
    print(json.dumps(result,sort_keys=True,ensure_ascii=False));return 0 if result['accepted'] else 2


if __name__=='__main__':raise SystemExit(main())
