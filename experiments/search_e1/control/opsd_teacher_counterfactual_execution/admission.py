"""CPU admission contract for an external root-owned budget/resource controller."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def checked(proof):
    path = Path(proof['path']).resolve(); before = path.stat()
    if sha(path) != proof['sha256']:
        raise ValueError('Admission source changed: ' + str(path))
    value = json.loads(path.read_text())
    if path.stat() != before: raise ValueError('Source changed during read')
    return value


def select_phase(records, phase, pilot_receipt=None, run_id=None):
    if len(records) != 64 or len({r['case_id'] for r in records}) != 64 or sum(r['pilot'] for r in records) != 16:
        raise ValueError('Need fixed unique64/16 preparation')
    if phase == 'pilot': return [r for r in records if r['pilot']]
    if phase != 'remaining': raise ValueError('Unknown phase')
    expected = {r['case_id'] for r in records if r['pilot']}
    if (pilot_receipt is None or pilot_receipt.get('accepted') is not True or pilot_receipt.get('run_id') != run_id or
            set(pilot_receipt.get('completed_case_ids', [])) != expected or len(pilot_receipt['completed_case_ids']) != 16 or
            pilot_receipt.get('state_reconstruction_accepted') is not True or pilot_receipt.get('null_effect_accepted') is not True):
        raise ValueError('Remaining48 need same-run accepted unique16 pilot')
    return [r for r in records if not r['pilot']]


def proc_identity(pid,proc_root='/proc'):
    base=Path(proc_root)/str(pid);before=(base/'stat').read_text().rsplit(') ',1)[1].split()
    status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (base/'status').read_text().splitlines() if ':' in line}
    uids=[int(x) for x in status['Uid'].split()]
    after=(base/'stat').read_text().rsplit(') ',1)[1].split()
    if len(uids)!=4 or before[19]!=after[19] or before[1]!=after[1]:raise ValueError('Kernel process identity changed')
    return dict(pid=pid,uid=uids[1],starttick=int(after[19]))


def validate_ticket(ticket, config, phase, worker_index, live=proc_identity):
    if ticket.get('schema') != 'm3-external-resource-ticket-v1' or ticket.get('accepted') is not True:
        raise ValueError('Root external controller ticket missing')
    if (ticket['run_id'], ticket['phase'], ticket['worker_index']) != (config['run_id'], phase, worker_index):
        raise ValueError('Resource ticket/run/phase/worker mismatch')
    if live(ticket['owner']['pid']) != ticket['owner'] or ticket['owner']['uid'] != os.getuid():
        raise ValueError('Resource controller identity no longer live')
    if ticket.get('m4_fully_released') is not True or ticket.get('root_lock_held') is not True:
        raise ValueError('M4 release/root-lock not admitted')
    if ticket['gpu_uuid'] not in config['approved_gpu_uuids']:
        raise ValueError('Foreign GPU admitted')
    if ticket['config_sha256'] != config['_config_sha256']:
        raise ValueError('Config identity differs')
    # External controller must bind its live evidence, not just booleans.
    for key in ('m4_release_proof', 'lock_proof', 'gpu_inventory_proof'):
        proof = checked(ticket[key])
        if key=='gpu_inventory_proof' and not 0<=__import__('time').time()-proof.get('sample_unix',0)<=30:
            raise ValueError('Resource inventory lease is stale/future')
        if proof.get('accepted') is not True or proof.get('run_id') != config['run_id']:
            raise ValueError('Root resource proof not accepted for run: ' + key)
    if live is proc_identity:
        environ = (Path('/proc') / str(ticket['owner']['pid']) / 'environ').read_bytes()
        if ('E1_RUN_ID=' + config['run_id']).encode() not in environ.split(b'\0'):
            raise ValueError('Resource owner private run environment differs')
    if ticket.get('expiry_unix', 0) <= __import__('time').time():
        raise ValueError('Resource admission ticket expired')


def validate_config(c):
    if c['schema'] != 'm3-fixed-counterfactual-v1' or c['workers'] != 3 or c['seed'] != 42:
        raise ValueError('Fixed M3 worker/seed configuration required')
    if c['retriever_url'] != 'http://10.120.17.101:18005/retrieve':
        raise ValueError('M3 requires admitted resident endpoint, no fallback')
    if set(c['adapters']) != {'source100', 'update1', 'update5', 'epoch1'} or not c['run_id']:
        raise ValueError('Need all four fixed adapters/same run')
    if len(set(c['approved_gpu_uuids'])) != 3:
        raise ValueError('Need three distinct externally reserved GPUs')


def fresh_runtime(config):
    reference=config['runtime_manifest']
    if reference['sha256']!='3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc':
        raise ValueError('Actual preparation3 runtime manifest identity differs')
    manifest=checked(reference)
    if len(manifest)!=305:raise ValueError('Need exact305 runtime source entries')
    root=Path(config['runtime']).resolve();proof={}
    for relative,digest in manifest.items():
        rel=Path(relative);path=(root/rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root):raise ValueError('Unsafe runtime path')
        before=path.stat()
        if sha(path)!=digest or path.stat()!=before:raise ValueError('Runtime source changed')
        proof[relative]=digest
    return proof


def fresh_retriever(config, admission, live=proc_identity, now=None):
    now=__import__('time').time() if now is None else now
    health=admission.check_live()
    expected=admission.expected
    if live(expected['pid'])!={k:expected[k] for k in ('pid','uid','starttick')}:
        raise ValueError('Candidate actual process identity changed')
    path=Path(config['retriever_observer_path']);before=path.stat();sample=json.loads(path.read_text())
    if path.stat()!=before:raise ValueError('Observer changed during read')
    if not 0<=now-sample['sample_end_unix']<=5 or sample.get('foreign_compute_apps_seen') or sample['current'].get('foreign_compute_apps'):
        raise ValueError('Resident observer stale/future/foreign history')
    for key in ('pid','uid','starttick'):
        if sample['candidate_'+key]!=expected[key]:raise ValueError('Observer candidate identity changed')
    if sample['source_identity_sha256']!=expected['source_identity_sha256'] or sample['physical_gpus']!=[0,1,2,3]:raise ValueError('Observer candidate source/GPU scope changed')
    observer=config['retriever_observer_identity']
    if live(observer['pid'])!=observer or sample['observer']['pid']!=observer['pid'] or sample['observer']['source_sha256']!=config['retriever_observer_source_sha256']:
        raise ValueError('Observer actual identity/source changed')
    return dict(health=health,observer_sample_unix=sample['sample_end_unix'],observer=observer,
                observer_summary_sha256=sha(path),foreign_compute_apps=[])


def validate_m4_pane(binding, raw, server_identity):
    import hashlib
    parts=raw.strip().split('\t')
    if len(parts)!=5 or parts[:3]!=['1','0',binding['pane_id']] or parts[3]!=str(binding['pane_pid']):
        raise ValueError('Actual preserved M4 pane is not bound terminal exit0')
    if hashlib.sha256(parts[4].encode()).hexdigest()!=binding['pane_start_command_sha256'] or server_identity!=binding['tmux_server']:
        raise ValueError('M4 tmux server/pane command identity changed')
    return dict(pane_dead=True,pane_dead_status=0,pane_id=parts[2],pane_pid=int(parts[3]),actual_command_sha256=binding['pane_start_command_sha256'])


def fresh_base_weights(manifest_path,model_path):
    manifest=json.loads(Path(manifest_path).read_text());source=manifest['source100'];root=Path(model_path).resolve()
    if Path(source['path']).resolve()!=root or len(source['files'])!=12 or len({x['relative_path'] for x in source['files']})!=12:
        raise ValueError('Need source100 exact12 inventory/path')
    proofs=[]
    for entry in source['files']:
        rel=Path(entry['relative_path']);path=(root/rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root):raise ValueError('Unsafe base weight path')
        before=path.stat();actual=sha(path)
        if actual!=entry['sha256'] or path.stat()!=before:raise ValueError('Actual fresh base weight SHA mismatch/change')
        proofs.append(dict(path=str(path),sha256=actual,bytes=before.st_size,inode=before.st_ino,device=before.st_dev,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns,fresh_cpu_content_hash=True))
    return proofs


def validate_m4_worker_identity(status, identity):
    actual=status['process']
    if status['status']!='COMPLETE' or any(actual[k]!=identity[k] for k in ('pid','uid','starttick')) or actual['uid']!=1005:
        raise ValueError('M4 actual nested process status identity mismatch')
