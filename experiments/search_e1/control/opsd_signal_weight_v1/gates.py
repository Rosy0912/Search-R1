"""CPU-only launch admission. Never kills jobs or treats a pending answer as consent."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen
import types


class Hold(RuntimeError):
    pass


def require(value, reason):
    if not value:
        raise Hold(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def bound(reference):
    require(isinstance(reference, dict) and reference.get('path') and reference.get('sha256'),
            'Missing explicit bound admission receipt')
    path = Path(reference['path']).resolve()
    before = path.stat()
    require(sha(path) == reference['sha256'], 'Receipt hash changed: ' + str(path))
    result = json.loads(path.read_text())
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Receipt changed during read')
    return result


def scoped(config,receipt):
    """Historical online200 permission cannot authorize this new four-signal-group goal."""
    scope=receipt.get('scope',{})
    require(scope.get('study_id')==config['study_id'],'Receipt is outside this new ablation study')
    if 'variant' in scope:
        require(scope['variant']==config['variant'] and scope.get('run_id')==config['run_id']
                and Path(scope.get('output_root','')).resolve()==Path(config['paths']['output_root']).resolve(),
                'Receipt does not bind this explicit variant/run/output')
    else:
        require(scope.get('variants')==['G_POS','G_NEG','H_POS','H_NEG']
                and scope.get('run_ids',{}).get(config['variant'])==config['run_id']
                and Path(scope.get('output_roots',{}).get(config['variant'],'')).resolve()==Path(config['paths']['output_root']).resolve(),
                'Study-wide receipt lacks all four exact group scopes')
    return receipt


def process(pid):
    root = Path('/proc') / str(pid)
    stat = (root / 'stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    require(fields[0] != 'Z', 'Zombie cannot satisfy ownership admission')
    status = (root / 'status').read_text().splitlines()
    uid_fields = next(line.split()[1:] for line in status if line.startswith('Uid:'))
    uid, euid = int(uid_fields[0]), int(uid_fields[1])
    require(uid == euid, 'Process real/effective UID differs')
    argv = (root / 'cmdline').read_bytes().split(b'\0')
    environment = dict(item.decode(errors='replace').split('=', 1)
                       for item in (root / 'environ').read_bytes().split(b'\0') if b'=' in item)
    after_stat = (root / 'stat').read_text()
    after_fields = after_stat[after_stat.rfind(')') + 2:].split()
    require(after_fields[19] == fields[19] and after_fields[0] != 'Z', 'Process changed during ownership read')
    after_uid = next(line.split()[1:3] for line in (root / 'status').read_text().splitlines()
                     if line.startswith('Uid:'))
    require(after_uid == uid_fields[:2], 'Process UID changed during ownership read')
    return dict(pid=int(pid), uid=uid, euid=euid, starttick=int(fields[19]), ppid=int(fields[1]),
                pgid=int(fields[2]), argv=[item.decode(errors='replace') for item in argv if item],
                run_id=environment.get('E1_RUN_ID'))


def same_process(actual, expected):
    return all(actual.get(key) == expected.get(key) for key in ('pid', 'uid', 'starttick'))


def inventory():
    def rows(kind, fields):
        result = subprocess.check_output(['nvidia-smi', '--query-' + kind + '=' + fields,
                                         '--format=csv,noheader,nounits'], text=True)
        return [[item.strip() for item in line.split(',')] for line in result.splitlines() if line.strip()]
    devices = [dict(index=int(row[0]), uuid=row[1], actual_free_mib=int(row[2]), total_mib=int(row[3]))
               for row in rows('gpu', 'index,uuid,memory.free,memory.total')]
    applications = [dict(gpu_uuid=row[0], pid=int(row[1]), used_mib=int(row[2]))
                    for row in rows('compute-apps', 'gpu_uuid,pid,used_gpu_memory')]
    return devices, applications


def owned_gpu_process(pid, run_id, owner, children, reader=process):
    """Follow the real live ancestry to a registered child, including PID start ticks."""
    anchors = {int(item['pid']): item for item in children}
    anchors[int(owner['pid'])] = owner
    actual = reader(pid)
    require(actual['uid'] == owner['uid'] and actual.get('run_id') == run_id,
            'GPU process is outside this explicit run/UID')
    visited = set()
    node = actual
    for _ in range(64):
        require(node['pid'] not in visited, 'Process ancestry loop')
        visited.add(node['pid'])
        if node['pid'] in anchors:
            require(same_process(node, anchors[node['pid']]), 'Registered PID was reused')
            return actual
        require(node['ppid'] > 1, 'GPU process has no registered live parent')
        node = reader(node['ppid'])
        require(node['uid'] == owner['uid'], 'GPU ancestry crosses another UID')
    raise Hold('GPU ancestry exceeds ownership bound')


def storage_admission(config, receipt, disk_reader=shutil.disk_usage, mount_reader=None):
    require(receipt.get('accepted') is True and receipt.get('cluster') == 'HKUST',
            'HKUST output storage admission has not been accepted')
    output = Path(config['paths']['output_root']).resolve()
    root = Path(receipt['output_root']).resolve()
    require(output == root, 'Storage receipt names another experiment output directory')
    anchor = output
    while not anchor.exists():
        require(anchor.parent != anchor, 'Output has no existing storage ancestor')
        anchor = anchor.parent
    mode = receipt.get('mode')
    if mode == 'same_cluster_shared':
        if mount_reader is None:
            def mount_reader(path):
                data = json.loads(subprocess.check_output(
                    ['findmnt', '-J', '-T', str(path), '-o', 'SOURCE,FSTYPE,TARGET'], text=True))
                return data['filesystems'][0]
        mounted = mount_reader(anchor)
        expected = receipt['mount']
        require(all(mounted.get(key) == expected.get(key) for key in ('source', 'fstype', 'target')),
                'Shared storage mount changed')
        require(mounted.get('fstype') in ('nfs', 'nfs4', 'lustre', 'ceph', 'gpfs', 'fuse.ceph'),
                'Local/bind storage does not prove same-cluster sharing')
        peer = receipt.get('cross_node_readback', {})
        require(peer.get('accepted') is True and peer.get('cluster') == 'HKUST'
                and peer.get('sha256') == receipt.get('probe_sha256'), 'Missing same-object peer readback')
        require(peer.get('hostname') and peer.get('hostname') != receipt.get('local_hostname'),
                'Readback must come from another explicitly authorized HKUST node')
        evidence = bound(peer.get('evidence'))
        require(evidence.get('accepted') is True and evidence.get('hostname') == peer['hostname']
                and evidence.get('sha256') == receipt['probe_sha256'], 'Peer readback evidence identity differs')
        probe = Path(receipt['probe_path']).resolve()
        require(sha(probe) == receipt['probe_sha256'], 'Actual shared probe changed after verification')
        require(receipt.get('retention_verified_until_unix', 0) > time.time(),
                'Shared server retention date is unknown or expired')
    elif mode == 'human_hkust_local_exception':
        authorization = receipt.get('human_authorization', {})
        require(authorization.get('explicit_exception') is True and authorization.get('text')
                and authorization.get('observed_unix', 0) > 0 and authorization.get('thread_id'),
                'No explicit human exception to shared-storage requirement')
        backup = receipt.get('backup_policy', {})
        require(backup.get('durable_target') and backup.get('synchronize_completed_checkpoints') is True,
                'Local exception requires a named durable checkpoint backup policy')
        require(not str(backup['durable_target']).startswith(('/Users/','/var/folders/'))
                and backup.get('large_outputs_remote_only') is True,
                'Weights/checkpoint/raw evidence must remain at the authorized HKUST remote target')
    else:
        raise Hold('Output storage answer is pending; cannot infer consent from restart request')
    usage = disk_reader(anchor)
    require(usage.free >= config['resources']['minimum_disk_bytes'], 'Output disk free space below admission threshold')
    return dict(mode=mode, realpath=str(output), existing_storage_anchor=str(anchor), disk_free_bytes=usage.free)


def publication_admission(config, receipt):
    require(receipt.get('accepted') is True and receipt.get('branch') == 'search-e1', 'Latest implementation was not published')
    commit = receipt.get('source_commit')
    require(isinstance(commit, str) and len(commit) == 40 and commit == receipt.get('remote_readback_sha'),
            'Missing actual push and remote SHA readback')
    require(config.get('source_commit') == commit, 'Configured source commit differs from publication')
    manifest = bound(config['helper_source_manifest'])
    require(receipt.get('source_manifest_sha256') == config['helper_source_manifest']['sha256'],
            'Published manifest is not this deployed helper tree')
    directory = Path(config['paths']['helper_root']).resolve()
    for name, entry in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe published source path')
        digest = entry['sha256'] if isinstance(entry, dict) else entry
        require(sha(directory / relative) == digest, 'Unpublished/hot-modified source: ' + name)
    expected_python = {name for name in manifest['files'] if name.endswith('.py')}
    actual_python = {str(path.relative_to(directory)) for path in directory.rglob('*.py')}
    require(expected_python == actual_python, 'Unpublished helper Python inventory changed')
    return dict(source_commit=commit, source_manifest_sha256=config['helper_source_manifest']['sha256'])


def verify_existing_joined_gate(gate):
    """Recompute the ORIGINAL K4 gate; it has no fabricated candidate fields."""
    require(gate.get('ntotal')==21015324 and gate.get('d')==768
            and gate.get('shard_bounds')==[[0,21015324]] and gate.get('tf32') is False,
            'Original oracle full-FP32 index/shape/TF32 contract changed')
    require(gate.get('score_atol')==1e-5 and gate.get('encoder')=='unchanged CPU FP32 E5'
            and gate.get('concurrent_512_requests')==2,'Original oracle tolerance/CPU encoder/concurrency changed')
    require(len(gate.get('queries',[]))==16 and len(gate.get('checks',[]))==16,
            'Original16-query acceptance is incomplete')
    matrices=[gate.get(k) for k in ('cpu_ids','gpu_ids','cpu_scores','gpu_scores')]
    require(all(isinstance(m,list) and len(m)==16 and all(isinstance(r,list) and len(r)==4 for r in m) for m in matrices),
            'Original oracle16x4 score/ID shape changed')
    cpuids,gpuids,cpus,gpus=matrices
    deltas=[]
    for index,(ci,gi,cs,gs) in enumerate(zip(cpuids,gpuids,cpus,gpus)):
        require(ci==gi and len(set(ci))==4 and all(type(x) is int and 0<=x<21015324 for x in ci),
                'Original oracle CPU/GPU document IDs differ')
        require(all(type(x) in (int,float) and math.isfinite(x) for x in cs+gs),'Nonfinite oracle score')
        require(all(cs[j]>=cs[j+1] and gs[j]>=gs[j+1] for j in range(3)),'Oracle topk scores are not ordered')
        delta=max(abs(a-b) for a,b in zip(cs,gs));check=gate['checks'][index]
        require(delta<=1e-5 and check.get('same_ids') is True and abs(check.get('max_score_delta',math.inf)-delta)<=1e-15,
                'Original gate numeric checks do not reconstruct')
        deltas.append(delta)
    return dict(queries=16,topk=4,recomputed_max_score_delta=max(deltas),score_atol=1e-5,
                scope='Existing original full-index K4 gate, not a new candidate gate')


def retriever_admission(config, receipt, getter=None):
    require(receipt.get('accepted') is True, 'Retriever admission is not accepted')
    url = config['paths']['retriever_url']
    require(receipt.get('retriever_url') == url, 'Retriever admission names another route')
    endpoint = urlparse(url)
    require(endpoint.port == 18003 and endpoint.path == '/retrieve', 'Four-card training requires the admitted HKUST4 GPU7 joined oracle route')
    require(endpoint.hostname == '10.123.4.11', 'Joined backend bind address changed; do not guess a loopback relay')
    gate_reference = dict(path=receipt['full_gpu_gate_path'], sha256=receipt['full_gpu_gate_sha256'])
    gate = bound(gate_reference)
    numeric=verify_existing_joined_gate(gate)
    health_url = urlunparse(endpoint._replace(path='/health', query='', fragment=''))
    if getter is None:
        def getter(url):
            with urlopen(url, timeout=10) as stream:
                return json.load(stream)
    health = getter(health_url)
    require(health.get('ready') is True and health.get('ntotal')==21015324 and health.get('d')==768
            and health.get('tf32') is False and health.get('query_micro')==64,'Existing K4 GPU oracle is not ready or its scientific shape changed')
    require(health.get('acceptance_sha256')==gate_reference['sha256']
            and health.get('source_manifest_sha256')==gate['source_manifest_sha256'],
            'Live K4 acceptance/source manifest differs from the original numeric gate')
    current = process(receipt['pid'])
    require(same_process(current, receipt), 'Retriever PID/UID/starttick changed')
    require(current['uid'] == config['owner']['uid'] == 1001,
            'K4 retriever is not the explicitly authorized sharing UID')
    scripts=[Path(arg).resolve() for arg in current['argv'] if arg.endswith('/retriever_stream.py')]
    require(len(scripts)==1,'Retriever is not the actual bound joined-service entry script')
    if receipt.get('source_path'):require(scripts[0]==Path(receipt['source_path']).resolve(),'Retriever actual entry path changed')
    require(sha(scripts[0])==receipt['source_identity_sha256'],'Retriever entry source bytes changed')
    return dict(health=health, owner=current,numeric_gate=numeric,entry_source_path=str(scripts[0]),
                entry_source_sha256=receipt['source_identity_sha256'])


def resource_limits(config, phase, devices, owned, owned_uuids):
    start_phases = {'sampling-start', 'training-start', 'evaluation-start', 'cpu-merge'}
    running_phases = {'sampling-running', 'training-running', 'evaluation-running'}
    require(phase in start_phases | running_phases, 'Unknown explicit resource lifecycle phase')
    if phase in start_phases:
        require(not owned, 'Previous owned GPU work is not released before the next stage')
    for device in devices:
        floor = (config['resources']['training_minimum_free_mib']
                 if phase in running_phases and device['uuid'] in owned_uuids
                 else config['resources']['startup_minimum_free_mib'])
        require(device['actual_free_mib'] >= floor, 'Actual GPU memory.free below phase/card-specific gate')


def memory_observer_admission(config):
    # A planned binding pathname avoids a config/observer-config-SHA cycle. Once
    # created, actual process and published source are verified on every gate.
    path=Path(config['gates']['memory_binding_path']).resolve()
    require(path.is_relative_to(Path(config['paths']['receipts']).resolve()),'Observer binding outside this group receipts')
    binding=bound(dict(path=str(path),sha256=sha(path)))
    require(binding.get('accepted') is True and binding.get('experiment_run_id')==config['run_id']
            and binding.get('config_sha256')==config['_config_sha256'],'Observer binding names another group/config')
    owner=binding['observer_identity'];actual=process(owner['pid'])
    require(same_process(actual,owner) and actual['uid']==config['owner']['uid']
            and actual.get('run_id')==owner['run_id'] and actual['run_id']!=config['run_id'],'Memory observer identity changed')
    source=Path(config['paths']['helper_root']).resolve()/'memory_observer.py'
    require(str(source) in actual['argv'],'Observer is not executing this helper source')
    manifest=bound(config['helper_source_manifest']);entry=manifest['files']['memory_observer.py']
    digest=entry['sha256'] if isinstance(entry,dict) else entry
    require(sha(source)==owner['source_sha256']==digest,'Observer source differs from publication')
    report_path=Path(binding['live_report_path']).resolve()
    require(report_path.is_relative_to(Path(config['paths']['receipts']).resolve()),'Observer report escaped group receipts')
    report=bound(dict(path=str(report_path),sha256=sha(report_path)))
    require(report['observer_identity']==owner and report['run_id']==config['run_id']
            and report['config_sha256']==config['_config_sha256'],'Live observer report provenance differs')
    age=time.time()-report['current']['sample_unix']
    require(0<=age<=30 and report['updated_unix']>=report['current']['sample_unix'],'Direct memory observer is stale/future-dated')
    require([d['uuid'] for d in report['current']['devices']]==config['owner']['gpu_uuids'],'Observer UUID changed')
    return dict(binding_path=str(path),observer_identity=owner,age_seconds=age,current=report['current'])


def scoped_cpu_processes(config, observer_optional=False):
    """Only freshly verified external CPU sources are explicit scanner exemptions."""
    path=Path(config['gates']['scoped_cpu_processes_path']).resolve()
    require(path==Path(config['paths']['receipts']).resolve()/'scoped-cpu-processes.json','CPU role registry path changed')
    reference=dict(path=str(path),sha256=sha(path));receipt=bound(reference)
    require(receipt.get('schema')=='opsd-ablation-scoped-cpu-processes-v1' and receipt.get('accepted') is True
            and receipt.get('run_id')==config['run_id'] and receipt.get('config_sha256')==config['_config_sha256'],
            'CPU role registry config/run changed')
    roles=receipt['roles'];names=[role['role'] for role in roles]
    require(len(names)==len(set(names)) and (set(names) in ({'study_driver','memory_observer'},{'memory_observer'})
            or observer_optional and set(names) in ({'study_driver'},set())),'Required exact CPU roles absent or duplicated')
    devices,apps=inventory();gpu_pids={app['pid'] for app in apps};identities=[];proofs=[]
    for role in roles:
        identity=role['identity'];actual=process(identity['pid'])
        require(all(actual.get(key)==identity.get(key) for key in ('pid','uid','starttick','ppid','pgid','argv','run_id'))
            and actual['uid']==config['owner']['uid'] and actual.get('run_id')
            and actual['run_id']!=config['run_id'] and role.get('no_gpu_or_training') is True,
            'External CPU role live identity/marker changed')
        argv_sha=hashlib.sha256(json.dumps(actual['argv'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        require(role.get('argv_sha256')==argv_sha and actual['pid'] not in gpu_pids,'CPU role argv/GPU ownership changed')
        input_value=bound(role['input_config'])
        if role['role']=='study_driver':
            require(input_value['study_id']==config['study_id'],'Driver belongs to another study')
            helper_root=Path(input_value['helper_root']).resolve();name='study_runner.py';option='--study-manifest'
            manifest_ref=input_value['helper_source_manifest'];publication_ref=input_value['publication'];commit=input_value['source_commit']
            require(helper_root.name=='opsd_signal_weight_v1','Driver is outside published ablation evaluator namespace')
        else:
            require(input_value=={key:value for key,value in config.items() if not key.startswith('_')},
                    'Observer is not reading this immutable group config')
            helper_root=Path(config['paths']['helper_root']).resolve();name='memory_observer.py';option='--config'
            manifest_ref=config['helper_source_manifest'];publication_ref=config['gates']['publication'];commit=config['source_commit']
        require(role['helper_source_manifest']==manifest_ref and role['publication']==publication_ref,
                'CPU role source/publication differs from its actual bound input')
        publication=bound(publication_ref);manifest=bound(manifest_ref)
        require(publication.get('accepted') is True and publication.get('branch')=='search-e1'
            and publication.get('source_commit')==publication.get('remote_readback_sha')==commit
            and publication.get('source_manifest_sha256')==manifest_ref['sha256'],'CPU role source was not published/read back')
        source=helper_root/name;entry=manifest['files'][name];digest=entry['sha256'] if isinstance(entry,dict) else entry
        require(Path(role['source']['path']).resolve()==source and role['source']['sha256']==digest
            and sha(source)==digest and source.stat().st_size==role['source']['bytes'],'CPU role actual source bytes changed')
        argv=actual['argv'];prefix=[config['paths']['python']]
        require(argv[:len(prefix)]==prefix and (argv[1:2]==[str(source)] or argv[1:3]==['-B',str(source)]),
                'CPU role is not the exact published Python entry')
        require(argv.count(option)==1 and argv[argv.index(option)+1]==role['input_config']['path'],
                'CPU role actual input argv differs')
        identities.append(actual);proofs.append(dict(role=role['role'],identity=actual,source=role['source'],input_config=role['input_config']))
    require(len({identity['run_id'] for identity in identities})==len(identities), 'CPU roles must use distinct non-training markers')
    return dict(accepted=True,receipt=reference,identities=identities,roles=proofs,observer_optional=observer_optional,
                direct_gpu_inventory=devices,checked_unix=time.time())


def _require_owned_release(config, run_id, owner, known_children=None, external_driver=False):
    """Reuse the reviewed V2 full scan; only the exact live phase controller is exempt."""
    import importlib.util
    source = Path(config['paths']['helper_root']) / 'owned_process_audit.py'
    manifest=bound(config['helper_source_manifest']);entry=manifest['files']['owned_process_audit.py']
    expected=entry['sha256'] if isinstance(entry,dict) else entry
    raw=source.read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,'Ownership scanner source SHA changed')
    module=types.ModuleType('ablation_bound_ownership_v2');module.__file__=str(source)
    exec(compile(raw,str(source),'exec'),module.__dict__)
    require(same_process(module.proc_snapshot(owner['pid']), owner), 'Release controller identity changed')
    def environment_reader(pid):
        value = module.proc_environment(pid)
        if pid == owner['pid']:
            require(same_process(module.proc_snapshot(pid), owner), 'Release controller PID was reused')
            # audit_same_uid deliberately gives owned matches precedence over
            # ignored_exact. Exempt only this exact expected-live supervisor;
            # all worker environments and ancestry remain visible.
            value = dict(value)
            value.pop('E1_RUN_ID', None)
        return value
    roots = [str(Path(config['paths']['helper_root']) / filename)
             for filename in ('sampling.py', 'trainer.py', 'teacher_diag.py', 'evaluation.py', 'evaluation_worker.py')]
    roots.append(str(Path(config['paths']['output_root']).resolve()))
    references = config.get('gates', {}).get('process_exclusions', [])
    cpu_roles=scoped_cpu_processes(config)
    if external_driver:
        driver=next(role['identity'] for role in cpu_roles['roles'] if role['role']=='study_driver')
        require(same_process(owner,driver) and owner.get('run_id')==driver['run_id'],
                'External release owner is not the exact admitted study driver')
        anchors=known_children or []
    else:
        require(owner.get('run_id')==run_id,'Training release owner marker differs')
        anchors=[owner]
    result = module.audit_same_uid(run_id, owner['uid'], owner['starttick'], known_children or [], roots,
                                  exclusions=module.load_exclusions(references), ignored_exact=[owner]+cpu_roles['identities'],
                                  environment_reader=environment_reader, ancestry_anchors=anchors)
    require(result['accepted_no_owned_or_opaque'], 'Owned/opaque descendant remains; preserve lock and inspect')
    require(same_process(module.proc_snapshot(owner['pid']), owner), 'Release controller changed after full scan')
    return dict(accepted=True, checked_unix=time.time(), audit=result, scoped_cpu_processes=cpu_roles,no_signals_sent=True)


def require_owned_release(config, run_id, owner, known_children=None):
    return _require_owned_release(config,run_id,owner,known_children)


def require_external_study_release(config, run_id, owner, known_children=None):
    """Driver's observer is external; known phase child ancestry still blocks."""
    return _require_owned_release(config,run_id,owner,known_children,external_driver=True)


def _require_launch(config, phase, run_id, owner, known_children=None, observer_bootstrap=False):
    gates = config.get('gates', {})
    require(os.getuid() == owner['uid'] and same_process(process(owner['pid']), owner), 'Controller owner identity changed')
    require(config.get('schema')=='opsd-signal-weight-config-v1' and config['run_id']==run_id,'Wrong ablation config/run')
    publication = publication_admission(config, scoped(config,bound(gates.get('publication'))))
    storage = storage_admission(config, scoped(config,bound(gates.get('storage'))))
    transition = scoped(config,bound(gates.get('resource_transition')))
    require(transition.get('accepted') is True and transition.get('cluster') == 'HKUST',
            'Missing explicit new-goal four-GPU transition receipt')
    retrieval = retriever_admission(config, bound(gates.get('retriever_admission')))
    cpu_roles=scoped_cpu_processes(config,observer_optional=observer_bootstrap)
    observer=(dict(startup_not_yet_spawned=True) if observer_bootstrap else memory_observer_admission(config))
    all_devices, all_applications = inventory()
    require(config['owner']['uid']==1001 and config['owner']['physical_gpus']==[4,5,6,7],
            'Signal weighting only owns the four authorized HKUST4 GPUs')
    devices=[item for item in all_devices if item['index'] in config['owner']['physical_gpus']]
    require([item['index'] for item in devices] == [4,5,6,7], 'Expected exactly four owned physical GPUs')
    require([item['uuid'] for item in devices] == transition.get('gpu_uuids'), 'Physical GPU UUID mapping changed')
    require([item['uuid'] for item in devices]==config['owner']['gpu_uuids'],'GPU UUID differs from immutable group config')
    owned_gpu_uuids=set(config['owner']['gpu_uuids'])
    applications=[app for app in all_applications if app['gpu_uuid'] in owned_gpu_uuids]
    children = known_children or []
    owned = []
    owned_uuids = set()
    for app in applications:
        require(app['pid']!=owner['pid'],'CPU-only controller holds an unexpected CUDA context')
        if app['pid'] == retrieval['owner']['pid']:
            require(app['gpu_uuid'] == devices[-1]['uuid'] and app['used_mib'] <= 1024, 'GPU7 oracle exceeded its sharing admission')
        else:
            owned.append(owned_gpu_process(app['pid'], run_id, owner, children))
            owned_uuids.add(app['gpu_uuid'])
    resource_limits(config, phase, devices, owned, owned_uuids)
    return dict(accepted=True, checked_unix=time.time(), phase=phase, run_id=run_id,
                publication=publication, storage=storage, retriever=retrieval,
                devices=devices, owned_gpu_processes=owned,memory_observer=observer,scoped_cpu_processes=cpu_roles,no_signals_sent=True)


def require_launch(config, phase, run_id, owner, known_children=None):
    return _require_launch(config,phase,run_id,owner,known_children)


def require_launch_without_memory(config, run_id, owner, known_children=None):
    """Only the CPU observer bootstrap; all actual GPU phase gates require memory."""
    require(not known_children,'Observer bootstrap cannot borrow running child GPU ownership')
    result=_require_launch(config,'cpu-merge',run_id,owner,[],observer_bootstrap=True)
    result['phase']='observer-startup'
    return result
