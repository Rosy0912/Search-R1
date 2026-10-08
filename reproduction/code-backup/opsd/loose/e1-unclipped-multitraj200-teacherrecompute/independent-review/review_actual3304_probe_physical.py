"""Local-only supplemental audit of actual shallow probe provenance originals."""
import csv
import datetime
import hashlib
import io
import json
import math
import pathlib
import re
import sys
import types
from unittest import mock

H = pathlib.Path(__file__).resolve().parents[1]
C = H / 'job3304-complete-probe-originals'
S = H / 'job3304-snapshot-3'
REPO = H.parent / 'opsd-unclipped-multitraj200-teacherrecompute'
SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
RUN = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304'
FIELD = 'short_native_teacher_logits_recompute_equivalence'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def original(name):
    return unique((C / name).read_bytes())


collector = H / 'collect_probe_originals.py'
raw_collector = collector.read_bytes()
need(sha(raw_collector) == 'ea9a1a77dee8bad4f2f657061eb883dc03e6233edf245276cb85860abf66fc62', 'Reviewed collector bytes')
fake_transport = types.ModuleType('transport')
fake_transport.remote = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('No remote in independent review'))
namespace = {'__name__': 'independent_read_only_decode', '__file__': str(collector)}
with mock.patch.dict(sys.modules, {'transport': fake_transport}):
    exec(compile(raw_collector, str(collector), 'exec'), namespace)
request = namespace['local_request'](H / namespace['BINDING_NAME'])
need(request['job_id'] == '3304' and request['source_commit'] == SOURCE and request['run_root'] == RUN,
     'Actual submission/V5 join')
raw_transport = (C / 'transport.stdout.original').read_bytes()
wire, bodies = namespace['decode_snapshot'](raw_transport, request)
exit_record = original('transport-exit.original.json')
need(type(exit_record['returncode']) is int and exit_record['returncode'] == 0 and
     sha(raw_transport) == exit_record['stdout_sha256'] and
     (C / 'transport.stderr.original').read_bytes() == b'' and
     exit_record['stderr_sha256'] == sha(b''), 'Actual successful transport originals')
inventory_raw = (C / 'inventory.json').read_bytes()
need(sha(inventory_raw) == '3f03ccd612498ad013571d4963b3fe3e540b2c3c0035f1f3cc590ea779879907', 'Actual inventory pin')
inventory = unique(inventory_raw)
need(len(bodies) == len(inventory['files']) == 52 and wire['missing_optional'] == [], 'All52 actual bounded originals')
need(inventory['collector_source_sha256'] == sha(raw_collector) and inventory['local_request'] == request,
     'Inventory collector/current local request join')
pins = {}
inodes = set()
for name, raw in bodies.items():
    metadata = inventory['files'][name]
    need((C / name).read_bytes() == raw and metadata ==
         {key: value for key, value in wire['files'][name].items() if key != 'base64'},
         'Exact actual wire/inventory/retained bytes: ' + name)
    need(metadata['physical'] == RUN + '/' + name and metadata['uid'] == 0 and
         len(raw) == metadata['bytes'] and sha(raw) == metadata['sha256'], 'Exact physical UID0 member')
    identity = (metadata['device'], metadata['inode'])
    need(identity not in inodes, 'Unexpected physical inode alias')
    inodes.add(identity)
    pins[name] = sha(raw)
pre = unique((H / 'ACTUAL_JOB3304_PRE_GPU_REVIEW.json').read_bytes())
need(sha((H / 'ACTUAL_JOB3304_PRE_GPU_REVIEW.json').read_bytes()) ==
     'f581fd1d02c8b349a141c51f7fbda1b1e42f1dcd7d5fe8bfefd552c5afff39a4', 'Prior actual pre-GPU review pin')
for name, pin in pre['original_files_sha256'].items():
    if name in pins:
        need(pins[name] == pin, 'Actual pre-GPU immutable original changed: ' + name)
binding = original('bindings/manifest.json')
for role, pin in pre['all_seven_binding_hashes'].items():
    need(binding['files'][role] == {'path': RUN + '/bindings/' + role + '.json', 'sha256': pin} and
         pins['bindings/' + role + '.json'] == pin, 'Seven-role binding unchanged')
gate_path = REPO / 'search_r1/baselines/checkpoint_gate.py'
gate_raw = gate_path.read_bytes()
need(sha(gate_raw) == '09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16', 'Reviewed gate bytes')
gate = {'__name__': 'independent_public_rsa_only', '__file__': str(gate_path)}
exec(compile(gate_raw, str(gate_path), 'exec'), gate)
envelope = original('bindings/gpu-init-approval.json')
gate['verify_rsa_signature'](envelope['payload'], envelope['signature_hex'], original('bindings/public.json'))
payload = envelope['payload']
for key in ('job_id', 'run_root', 'source_commit', 'nonce', 'ready_sha256', 'binding_sha256',
            'public_key_sha256', 'process_start_ticks'):
    expected = SOURCE if key == 'source_commit' else RUN if key == 'run_root' else pre.get(key)
    need(payload[key] == expected, 'Actual init-only signed identity: ' + key)
need(payload['authorization'] == 'EXPLICIT_OPSD_E1_GPU_INITIALIZATION_ONLY' and
     payload['pid'] == pre['wrapper_pid'] and payload['uid'] == 0 and
     payload['independent_review']['sha256'] == sha((H / 'ACTUAL_JOB3304_PRE_GPU_REVIEW.json').read_bytes()),
     'Actual signed independent initialization admission')

probe_review_raw = (H / 'ACTUAL3304_GPU_RECOMPUTE_CAPACITY_INDEPENDENT_REVIEW.json').read_bytes()
need(sha(probe_review_raw) == '0e8f4d2ccf99ebd6fb22d235a231ef85b54bfbea8d61a36609c5aa7c1962955e', 'Prior numeric review pin')
probe_review = unique(probe_review_raw)
for name in ['evidence/capacity-micro4/micro-4.json', 'evidence/capacity-micro4/torchrun.log',
             'evidence/capacity-micro4/exit.json', 'evidence/hardware-exit.json']:
    need(pins[name] == probe_review['original_files_sha256'][name], 'Actual completed probe original unchanged')
aggregate = original('evidence/capacity-micro4/micro-4.json')
logged = {}
for line in (C / 'evidence/capacity-micro4/torchrun.log').read_text().splitlines():
    if line.startswith('{'):
        row = unique(line)
        need(type(row['rank']) is int and row['rank'] not in logged, 'Actual raw rank uniqueness')
        logged[row['rank']] = row
need(set(logged) == set(range(8)), 'Actual eight raw probe records')
standalone = []
for rank in range(8):
    current = original('evidence/capacity-micro4/micro-4-rank-' + str(rank) + '.json')
    need(current == logged[rank], 'Standalone actual rank vs full raw console exact')
    record = next(item for item in aggregate['ranks'] if item['rank'] == rank)
    need(all(current[key] == value for key, value in record.items()) and
         set(current) - set(record) == {'elapsed_seconds'}, 'Exact aggregate fields and producer-finally elapsed suffix')
    numeric = original('evidence/capacity-micro4/teacher-logits-recompute-rank-' + str(rank) + '.json')
    lp = original('evidence/capacity-micro4/old-logprob-rank-' + str(rank) + '.json')
    need(numeric == record[FIELD] and lp == record['old_logprob_capacity'], 'Standalone numeric/LP original joins')
    standalone.append(dict(rank=rank, rank_original_sha256=pins['evidence/capacity-micro4/micro-4-rank-' + str(rank) + '.json'],
        short_original_sha256=pins['evidence/capacity-micro4/teacher-logits-recompute-rank-' + str(rank) + '.json'],
        lp_original_sha256=pins['evidence/capacity-micro4/old-logprob-rank-' + str(rank) + '.json']))

hardware = []
hardware_pids = set()
uuids = pre['uuids']
for rank in range(8):
    name = 'evidence/hardware-rank-' + str(rank) + '.json'
    row = original(name)
    need(all(type(row[key]) is int for key in ('rank', 'local_rank', 'world_size', 'uid', 'pid')) and
         row['rank'] == row['local_rank'] == rank and row['world_size'] == 8 and row['uid'] == 0 and
         row['pid'] > 0 and row['pid'] not in hardware_pids and row['source_commit'] == SOURCE and
         row['job_id'] == '3304' and re.search(r'(?:^|/)job_3304/step_0(?:\n|$)', row['cgroup'], re.M) and
         row['actual_cuda_uuid'] == row['uuid'] == uuids[rank] and
         row['gpu_name'] == 'NVIDIA H100 80GB HBM3' and row['torch'] == '2.4.0+cu121' and
         row['flash_finite'] is True and row['allreduce_value'] == sum(range(1, 9)),
         'Actual native8 hardware/NCCL/Flash and physical UUID binding')
    hardware_pids.add(row['pid'])
    hardware.append(dict(rank=rank, pid=row['pid'], uuid=row['uuid'], original_sha256=pins[name]))
need(original('evidence/hardware-exit.json') == {'returncode': 0}, 'Actual hardware exit')
hardware_stdout_bytes = len(bodies['evidence/hardware-torchrun.log'])

service = original('evidence/service.json')
http = original('evidence/http-top3-24.json')
owner = service['owner']
need(type(owner['pid']) is int and owner['pid'] == 2543106 and type(owner['uid']) is int and owner['uid'] == 0 and
     type(owner['start']) is str and owner['start'] == '1834765439' and
     re.search(r'(?:^|/)job_3304/step_0(?:\n|$)', owner['cgroup'], re.M), 'Actual own service identity')
need(all(http['service'][key] == owner[key] for key in ('pid', 'start', 'uid', 'cgroup')) and
     http['job_id'] == '3304' and set(service['uuids']) == set(uuids) and len(service['uuids']) == 8,
     'Actual local retrieval/API process and physical UUIDs')
bound = unique((H / 'runtime-tools-bound-fcecbea5d939-v5.json').read_bytes())
expected_command = [
    '/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python', '-u',
    bound['formal']['remote_directory'] + '/retriever_entry.py',
    '--index_path', '/mnt/data/shaoshitong/search-r1-rl/opsd/assets-root/assets/wiki/e5_Flat.index',
    '--corpus_path', '/mnt/data/shaoshitong/search-r1-rl/opsd/assets-root/assets/wiki/wiki-18.jsonl',
    '--retriever_model', '/mnt/data/shaoshitong/search-r1-rl/opsd/assets-root/assets/e5-base-v2',
    '--device', 'cuda', '--faiss_gpu', '--host', '127.0.0.1', '--port', '18084', '--threads', '8', '--topk', '3']
need(service['command'] == expected_command and
     service['pythonpath'] == bound['remote_repo'] + ':/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib',
     'Actual pinned local CUDA E5/GPU FAISS/Wiki command')
queries = http['queries']
responses = http['response']['result']
need(len(queries) == len(responses) == 24 and len(set(queries)) == 8 and queries[:8] * 3 == queries and
     type(http['seconds']) in (int, float) and math.isfinite(http['seconds']) and http['seconds'] > 0,
     'Actual24 API requests/eight unique queries/timing')
for row in responses:
    need(len(row) == 3, 'Actual top3 response')
    for item in row:
        need(type(item['document']['contents']) is str and bool(item['document']['contents']) and
             type(item['document']['id']) is str and bool(item['document']['id']) and
             type(item['score']) in (int, float) and math.isfinite(item['score']), 'Actual retrieved content and finite score')
# Snapshot3 has unfiltered NVML applications after the probe. Join only the
# actually bound service; training-worker PID/cgroup proof is a later layer.
snapshot_wire = unique((S / 'transport.stdout.original').read_bytes())
app_rows = list(csv.reader(io.StringIO(snapshot_wire['live']['gpu_apps']['stdout'])))
service_gpu_uuids = [row[0].strip() for row in app_rows if row[1].strip() == str(owner['pid'])]
need(len(service_gpu_uuids) == 8 and set(service_gpu_uuids) == set(uuids), 'Actual service NVML occupancy on own8 GPUs')
retriever_log = bodies['retriever.log'].decode('utf-8')
need('Started server process [2543106]' in retriever_log and
     'http://127.0.0.1:18084' in retriever_log and 'POST /retrieve HTTP/1.1" 200 OK' in retriever_log,
     'Actual loopback service log witness')
report = dict(version=1, status='PASS_ACTUAL_PROBE_PHYSICAL_HARDWARE_RETRIEVAL_ORIGINALS_ONLY',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), job_id='3304', source_commit=SOURCE,
    captured_utc=inventory['observed_utc'], original_capsule_path=str(C),
    original_inventory_sha256=sha(inventory_raw), original_transport_sha256=sha(raw_transport),
    collector_sha256=sha(raw_collector), collector_software_review_sha256=
        sha((H / 'COLLECT_PROBE_ORIGINALS_INDEPENDENT_SOFTWARE_REVIEW.json').read_bytes()),
    prior_numeric_probe_review_sha256=sha(probe_review_raw), original_files_sha256=pins,
    actual_original_count=52, missing_optional_count=0, all52_physical_UID0_stable_or_prefix_metadata_join=True,
    actual_init_Rsa_signature_and_source_ready_identity_verified=True,
    actual_init_envelope_sha256=pins['bindings/gpu-init-approval.json'],
    actual_standalone_rank_numeric_lp_joins=standalone, actual_hardware_rank_records=hardware,
    actual_hardware_allreduce_value=36., actual_hardware_exit=0,
    hardware_torchrun_original_bytes=hardware_stdout_bytes,
    actual_retrieval=dict(service_pid=owner['pid'], service_start_ticks=owner['start'], service_cgroup=owner['cgroup'],
        service_source_sha256=pins['evidence/service.json'], http_original_sha256=pins['evidence/http-top3-24.json'],
        loopback_url='http://127.0.0.1:18084/retrieve', gpu_E5_gpu_FAISS_command=True,
        same_node_own_eight_UUIDs_and_NVML_service_identity=True, topk=3, submitted_queries=24,
        unique_queries=8, actual_nonempty_document_results=72, client_end_to_end_seconds=http['seconds'],
        latency_scope='Client wall time surrounds loopback request plus service encode/index search/response and client decode; not network-only, component exclusive GPU timing or full rollout latency.'),
    scope='Supplemental shallow original/provenance join for completed numeric/capacity probe. This does not upgrade to first5, actual trajectory content/state restore, effects, terminal closure or final200.',
    limits=['Immutable JSON observed same-FD stable full bytes; .log files retain verified initial-length prefixes and are not closed lifetime logs.',
        'Hardware stdout is actually empty; eight physical rank JSON and hardware exit0 are the direct evidence, with no invented console rows.',
        'All24 standalone capacity/numeric/LP JSON exactly join the separately reviewed raw rank/aggregate statistics; raw gradient/logit tensor bytes are not in this shallow capsule.',
        'Capacity rank cgroup/host are recorded by native probe, while per-rank hardware PIDs belong to the separate small hardware step. They are not claimed to be the capacity or training PIDs.',
        'Pinned model/data/tool original binding and pre-ready validator path are retained; no local large model/Arrow/corpus/index re-read was performed by this reviewer.',
        'Service JSON/HTTP/NVML establish actual same-node CUDA E5/GPU-FAISS loopback identity and top3 API conformance; retrieval quality/algorithm training correctness are separate.',
        'Initial reviewer expected run_root in the prior preGPU review, whose schema omits that field; original script/new local rerun failure retained. Current check joins the actual request/wire/ready/envelope RUN instead. Only local exact decoding, source-defined public RSA verification and cross-file mathematics were run. No SSH, transport, CUDA, job control, signing, private key, source edit or production main.'])
p = H / 'ACTUAL3304_PROBE_PHYSICAL_INFRA_ORIGINALS_INDEPENDENT_REVIEW.json'
with p.open('x') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write('\n')
md = H / 'ACTUAL3304_PROBE_PHYSICAL_INFRA_ORIGINALS_INDEPENDENT_REVIEW.md'
with md.open('x') as handle:
    handle.write('All52 actual shallow originals join their transported fullSHA, physical UID0 path and same-FD metadata; no optional member is missing. Seven pre-GPU bindings and the actual init-only RSA envelope join the prior independent review. Each eight-rank standalone capacity, short numerical and LP report matches the original raw GPU record/aggregate reviewed separately.\n\nEight small hardware rank files prove their own h100-10/job3304 step0 PIDs, physical H100 UUIDs, nativeTorch2.4, finite FlashAttention and NCCL SUM36; hardware exit0 joins. Hardware stdout is genuinely empty and is not filled with inferred records. These hardware PIDs are not called capacity/training PIDs.\n\nRetrieval servicePID2543106/start1834765439 is bound to the same job/node and all eight allocated UUIDs, also seen in raw NVML applications. The actual command uses CUDA E5/GPUFAISS, the pinned Wiki2018 index and127.0.0.1:18084/top3. All24 queries return72 nonempty documents with finite scores. The0.672572-second timer measures the complete local client request/service/response/decode interval, not network-only time.\n\nThis closes the supplemental probe infrastructure/provenance scope only. Live log prefixes are not lifetime closure; first5 saved state/HF/trajectory content, training-worker identities/resource history, effect and final200 remain separate. No remote, GPU, signing, private-key, job control or production execution occurred.\n')
print(json.dumps(dict(status=report['status'], json_sha256=sha(p.read_bytes()), md_sha256=sha(md.read_bytes()),
                     original_count=52, retrieval_seconds=http['seconds'])))
