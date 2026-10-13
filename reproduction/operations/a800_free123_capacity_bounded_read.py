"""Read one recorded capacity holder; verify actual identity, UUIDs and bytes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def small(path):
    raw = path.read_bytes()
    require(len(raw) < 65536, 'bounded record exceeded')
    return raw


def read_identity(expected):
    path = Path('/proc') / str(expected['pid'])
    try:
        raw = (path / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        uid = next(x for x in (path / 'status').read_text().splitlines() if x.startswith('Uid:')).split()
        current = {'pid': expected['pid'], 'start_ticks': int(fields[19]), 'sid': int(fields[3]),
                   'pgid': int(fields[2]), 'uid': int(uid[1]), 'euid': int(uid[2]), 'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None
    current['birth_matches'] = all(current[k] == expected[k] for k in ('pid', 'start_ticks', 'sid', 'pgid', 'uid', 'euid'))
    current['alive'] = current['birth_matches'] and current['state'] not in ('Z', 'X')
    return current


def epoch(value):
    return datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


require(sys.flags.optimize == 0, 'Optimization disabled required')
p = json.load(sys.stdin)
indices = [1, 2, 3]
amount = 60 * 1024 ** 3
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
host = socket.gethostname()
require(p['boot_id'] == boot and p['hostname'] == host, 'Receipt boot/node mismatch')
require(p['physical_indices'] == indices and p['bytes_per_gpu'] == amount, 'Receipt capacity mismatch')
require(p['capacity_exclusivity'] == 'NONEXCLUSIVE' and p['native_training_admission'] is False, 'Receipt is not a capacity holder')
require(p['launcher_optimize'] == 0, 'Launcher guard was optimized')
namespace = Path(p['namespace'])
info = namespace.stat()
require(namespace.resolve() == namespace and info.st_uid == p['identity']['uid']
        and info.st_mode & 0o777 == 0o700, 'Namespace path/ownership/mode mismatch')
raw = small(namespace / 'reservation.json')
state = json.loads(raw)
require(state['role'] == 'only_capacity_reservation' and state['namespace'] == p['namespace'], 'State namespace/role mismatch')
require(state['namespace_device'] == info.st_dev and state['namespace_inode'] == info.st_ino, 'Namespace changed')
require(state['source_sha256'] == p['source_sha256'] and state['source_path'] == p['source_path'], 'State source mismatch')
require(state['source_backup_sha256'] == p['source_sha256'], 'Backup declaration mismatch')
require(state['physical_indices'] == indices and state['bytes_per_gpu'] == amount, 'State capacity mismatch')
require(state['capacity_exclusivity'] == 'NONEXCLUSIVE' and state['native_training_admission'] is False, 'State admission scope mismatch')
require(state['identity']['boot_id'] == boot and state['identity']['hostname'] == host, 'State boot/node mismatch')
for key in ('pid', 'start_ticks', 'sid', 'pgid', 'uid', 'euid'):
    require(state['identity'][key] == p['identity'][key], 'State identity mismatch: ' + key)
claim_raw = small(namespace / 'claim.json')
claim = json.loads(claim_raw)
require(claim['role'] == state['role'] and claim['namespace'] == p['namespace']
        and claim['physical_indices'] == indices and claim['source_sha256'] == p['source_sha256'], 'Claim mismatch')
require(claim['identity'] == state['identity'], 'Claim identity mismatch')
backup = namespace / 'source.py'
bi = backup.stat()
require(not backup.is_symlink() and bi.st_nlink == 1 and bi.st_uid == info.st_uid
        and bi.st_mode & 0o777 == 0o400, 'Source backup ownership/mode mismatch')
backup_sha = hashlib.sha256(backup.read_bytes()).hexdigest()
require(backup_sha == p['source_sha256'], 'Actual source backup mismatch')

inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=30)
mapping = {}
for row in inventory.splitlines():
    fields = [x.strip() for x in row.split(',')]
    mapping[fields[1].lower()] = {'index': int(fields[0]), 'uuid': fields[1].lower(), 'pci_bus_id': fields[2],
                                'used_mib': int(fields[3]), 'util_percent': int(fields[4])}
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                               '--format=csv,noheader,nounits'], text=True, timeout=30)
owned = []
foreign = []
auxiliary = []
for row in apps.splitlines():
    fields = [x.strip() for x in row.split(',')]
    uuid = fields[0].lower()
    item = {'gpu_index': mapping[uuid]['index'], 'pid': int(fields[1]), 'used_mib': int(fields[2]), 'gpu_uuid': uuid}
    if item['pid'] == p['identity']['pid']:
        owned.append(item)
    elif item['gpu_index'] in [0, 1, 2, 3]:
        foreign.append(item)
    elif item['gpu_index'] in [5, 6]:
        auxiliary.append(item)
for item in foreign + auxiliary:
    proc = Path('/proc') / str(item['pid'])
    try:
        text = (proc / 'stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        item.update(start_ticks=int(fields[19]), sid=int(fields[3]), uid=proc.stat().st_uid, state=fields[0],
                    cwd_sha256=hashlib.sha256(os.readlink(proc / 'cwd').encode()).hexdigest())
    except (FileNotFoundError, ProcessLookupError):
        item['proc_missing'] = True
current = read_identity(p['identity'])
verified = False
post_ready_verified = False
if state['status'] == 'READY':
    require(current is not None and current['alive'], 'READY holder no longer has the same live birth')
    visible = state['visible_mapping']
    require(len(visible) == 3, 'Visible mapping count mismatch')
    uuids = {}
    for ordinal, physical in enumerate(indices):
        entry = visible[ordinal]
        uuid = entry['uuid'].lower()
        require(entry['cuda_ordinal'] == ordinal and entry['physical_index'] == physical
                and entry['requested_bytes'] == amount and mapping[uuid]['index'] == physical, 'Physical UUID/ordinal mismatch')
        uuids[physical] = uuid
    require(len(set(uuids.values())) == 3, 'Repeated physical UUID')
    preflight = p['actual_preflight']
    require(preflight['selected_compute_apps'] == [] and len(preflight['gpus']) == 3, 'Prelaunch was not free')
    for gpu in preflight['gpus']:
        require(gpu['index'] in indices and gpu['uuid'].lower() == uuids[gpu['index']]
                and gpu['pci_bus_id'].lower() == mapping[gpu['uuid'].lower()]['pci_bus_id'].lower()
                and gpu['free_mib'] >= 62464, 'Prelaunch mapping/headroom mismatch')
    require(sorted(x['index'] for x in preflight['gpus']) == indices, 'Prelaunch indices mismatch')
    require(sorted(x['gpu_index'] for x in owned) == indices, 'Owned current device count mismatch')
    for item in owned:
        require(item['gpu_uuid'] == uuids[item['gpu_index']] and item['used_mib'] * 1024 ** 2 >= amount,
                'Current owned UUID/bytes mismatch')

    def validate_sample(sample, allow_foreign=False):
        require(isinstance(sample, dict) and len(sample['gpus']) == 3, 'Missing bounded capacity sample')
        require(sorted(g['index'] for g in sample['gpus']) == indices, 'Sample device count mismatch')
        for gpu in sample['gpus']:
            require(gpu['uuid'].lower() == uuids[gpu['index']]
                    and gpu['pci_bus_id'].lower() == mapping[gpu['uuid'].lower()]['pci_bus_id'].lower(), 'Sample physical mapping mismatch')
        usage = {}
        for app in sample['compute_apps']:
            require(app['uuid'].lower() in uuids.values(), 'Sample contains an unselected UUID')
            if app['pid'] == p['identity']['pid']:
                uuid = app['uuid'].lower()
                usage[uuid] = usage.get(uuid, 0) + app['used_bytes']
            else:
                require(allow_foreign, 'Foreign context in READY admission sample')
        require(len(sample['torch_allocated_bytes']) == 3 and len(sample['torch_reserved_bytes']) == 3,
                'Sample Torch device count mismatch')
        for ordinal, physical in enumerate(indices):
            require(usage.get(uuids[physical], 0) >= amount and sample['torch_allocated_bytes'][ordinal] >= amount
                    and sample['torch_reserved_bytes'][ordinal] >= amount, 'Sample lacks full owned capacity')
        require(sample['capacity_exclusivity'] == 'NONEXCLUSIVE', 'Sample exclusive admission mismatch')

    ready = state['ready_samples']
    require(len(ready) == 2, 'Two real READY samples required')
    for sample in ready:
        validate_sample(sample)
    require(epoch(ready[1]['at']) >= epoch(ready[0]['at']) + 1, 'READY samples not separately measured')
    verified = True
    holding = state['holding_sample']
    if holding is not None:
        validate_sample(holding, allow_foreign=True)
        require(epoch(holding['at']) >= epoch(ready[1]['at']) and 0 <= time.time() - epoch(holding['at']) <= 90,
                'Holding sample is not a fresh post-READY measurement')
        post_ready_verified = True
    # The first reading may precede the first periodic sample; final acceptance needs this flag.
    current = read_identity(p['identity'])
    require(current is not None and current['alive'], 'Owned birth changed after inventory validation')

out = {'role': 'actual_owned_capacity_reservation_bounded_observation', 'checked_unix': time.time(),
       'process': current, 'state': state, 'record_sha256': hashlib.sha256(raw).hexdigest(),
       'owned_cuda_applications': owned, 'foreign_training_cuda_applications': foreign,
       'foreign_auxiliary_cuda_applications': auxiliary, 'actual_owned_60GiB_each_verified': verified,
       'post_READY_retention_sample_verified': post_ready_verified, 'birth_revalidated_after_NVML': True,
       'boot_id': boot, 'hostname': host, 'namespace_identity': {'device': info.st_dev, 'inode': info.st_ino,
                                                              'uid': info.st_uid, 'mode': oct(info.st_mode & 0o777)},
       'claim_sha256': hashlib.sha256(claim_raw).hexdigest(), 'source_backup_sha256': backup_sha,
       'capacity_exclusivity': 'NONEXCLUSIVE', 'native_training_admission': False,
       'signals_sent': False, 'trainer_started': False, 'native_algorithm_accepted': False,
       'reservation_peak_is_training_peak': False, 'reader_optimize': sys.flags.optimize}
print(json.dumps(out))
