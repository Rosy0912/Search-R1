"""Owner-run full GPU and HTTP equality gate. Does not activate/admit/switch.

The private compare endpoint compares the full original GPU scanner using the
same encoded query arrays and records actual GEMM shapes. Only one real
320-query request is sent sequentially to the shared original GPU7 HTTP oracle. All
shape tests and the 8 x 2560 stress scan stay on the candidate. External actual
whole-card/process-memory sample times must cover the complete gate interval.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
import math


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def post(url, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-E1-Probe-Token"] = token
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=7200) as response:
        return json.load(response)


def concurrent_requests(url, payloads, token=None):
    barrier = threading.Barrier(len(payloads))
    def one(payload):
        barrier.wait()
        started = time.monotonic()
        response = post(url, payload, token)
        return response, time.monotonic() - started
    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        return list(executor.map(one, payloads))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def live_process(pid):
    base = Path('/proc') / str(pid)
    fields = (base/'stat').read_text().rsplit(')',1)[1].split()
    uid = int(next(l.split()[1] for l in (base/'status').read_text().splitlines() if l.startswith('Uid:')))
    argv = (base/'cmdline').read_bytes().decode().split('\0')
    return dict(pid=pid,uid=uid,starttick=int(fields[19]),argv=[a for a in argv if a])


def verify_source(identity, health):
    for key in ['pid','uid','starttick','source_identity_sha256']:
        if identity[key] != health[key]:
            raise RuntimeError('Live candidate identity differs: ' + key)
    if {k:live_process(health['pid'])[k] for k in ['pid','uid','starttick']} != {k:health[k] for k in ['pid','uid','starttick']}:
        raise RuntimeError('Candidate process was replaced')
    measured = hashlib.sha256(json.dumps(identity['identity'],sort_keys=True).encode()).hexdigest()
    if measured != health['source_identity_sha256']:
        raise RuntimeError('Candidate identity digest was not recomputed from its manifest')
    for path, expected in identity['identity']['sources'].items():
        if file_sha(Path(path)) != expected:
            raise RuntimeError('Candidate source changed: ' + path)
    for key in ['original_scan','original_encoder','assets_manifest','asset_bindings']:
        entry=identity['identity'][key]
        if file_sha(Path(entry['path'])) != entry['sha256']:
            raise RuntimeError('Bound operator/config identity changed: ' + entry['path'])


def verify_oracle(health, receipt_path):
    expected=dict(pid=2239529,uid=1005,starttick=914664099)
    process=live_process(expected['pid'])
    if any(process[key] != value for key,value in expected.items()):
        raise RuntimeError('Original GPU7 oracle PID/UID/starttick changed')
    receipt=json.loads(receipt_path.read_text())
    sha='b2626bd30e2c9ca12b94f17d4464ade6e012bbef36d7c2dd051c36e4f14fb614'
    if health.get('pid') != expected['pid'] or health.get('source_identity_sha256') != sha:
        raise RuntimeError('Original GPU7 live oracle source identity changed')
    if receipt['pid'] != expected['pid'] or receipt['source_identity_sha256'] != sha:
        raise RuntimeError('Original GPU7 source receipt differs')
    if hashlib.sha256(json.dumps(receipt['identity'],sort_keys=True).encode()).hexdigest() != sha:
        raise RuntimeError('Original GPU7 source identity digest invalid')
    if '/data/shaoshitong/search-e1-retriever-segmented-20261007/source/server.py' not in process['argv']:
        raise RuntimeError('Original GPU7 executable path differs')
    for path,expected_sha in receipt['identity']['sources'].items():
        if file_sha(Path(path)) != expected_sha:raise RuntimeError('Original oracle source changed')
    return process


def verified_assets(path):
    evidence = json.loads(path.read_text())
    if evidence.get("accepted") is not True:
        raise RuntimeError("Asset/source identity evidence is not accepted")
    comparisons = evidence.get("comparisons", [])
    if not comparisons:
        raise RuntimeError("Actual index/corpus/encoder/source SHA comparisons are required")
    kinds = set()
    for item in comparisons:
        kind, left, right = item["kind"], item["k1_sha256"], item["k4_sha256"]
        if not isinstance(left, str) or len(left) != 64 or any(c not in "0123456789abcdef" for c in left):
            raise RuntimeError("Invalid SHA256 identity comparison")
        if left != right:
            raise RuntimeError("resident/original asset/source SHA differs: " + item.get("path", kind))
        kinds.add(kind)
    if not {"index", "corpus", "encoder", "scanner_source", "encoder_source"} <= kinds:
        raise RuntimeError("Evidence must cover full index, corpus, encoder and both operator sources")
    return evidence


def memory_covering(path, before, gate_start, gate_end):
    deadline = time.monotonic() + 15
    while True:
        try:
            memory = json.loads(path.read_text())
            start = float(memory["sample_start_unix"])
            end = float(memory["sample_end_unix"])
            updated = float(memory["updated_at_unix"])
        except (OSError, ValueError, KeyError, TypeError):
            memory = None
        if memory is not None and end >= gate_end:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Memory sampler has not supplied actual times through gate end")
        time.sleep(0.5)
    if not all(math.isfinite(value) for value in [start, end, updated]):
        raise RuntimeError("Invalid memory sample timestamps")
    if not start <= gate_start <= gate_end <= end <= updated <= time.time() + 2:
        raise RuntimeError("Actual memory sample start/end/update times do not cover the gate")
    if time.time() - updated > 30 or int(memory.get("samples", 0)) < 2:
        raise RuntimeError("Memory evidence is stale or has insufficient actual samples")
    if memory["candidate_pid"] != before["pid"]:
        raise RuntimeError("Memory evidence belongs to a different process")
    if memory.get('candidate_uid') != before['uid'] or memory.get('candidate_starttick') != before['starttick']:
        raise RuntimeError('Memory evidence PID was replaced or UID changed')
    if memory.get("source_identity_sha256") != before["source_identity_sha256"]:
        raise RuntimeError("Memory evidence must bind the candidate source identity")
    if memory.get("free_measurement") != "nvidia-smi memory.free":
        raise RuntimeError("Memory evidence must query actual memory.free, not compute total minus used")
    if memory.get("physical_gpus") != [0,1,2,3] or memory.get("gpu_uuids") != before["gpu_inventory_uuids"]:
        raise RuntimeError("Memory sampler must bind all four physical card UUIDs")
    if set(memory.get("max_process_by_uuid_mib", {})) != set(before["gpu_inventory_uuids"]):
        raise RuntimeError("Memory sampler omitted resident GPU process evidence")
    if 'foreign_compute_apps_seen' not in memory:
        raise RuntimeError('Memory sampler must retain historical foreign process evidence')
    if memory.get('foreign_compute_apps_seen') or memory.get("current", {}).get("foreign_compute_apps"):
        raise RuntimeError("Resident GPU has an unapproved compute process")
    if set(memory.get('minimum_actual_free_by_uuid_mib',{})) != set(before['gpu_inventory_uuids']):
        raise RuntimeError('Missing per-card direct free minima')
    if any(float(value) < 6144 for value in memory['minimum_actual_free_by_uuid_mib'].values()):
        raise RuntimeError('A resident card violated the direct free requirement')
    if any(not 0 < float(value) <= 18432 for value in memory['max_process_by_uuid_mib'].values()):
        raise RuntimeError('A resident card is absent or violated the per-card process cap')
    process_peak, free_min = float(memory["max_process_mib"]), float(memory["min_free_mib"])
    if not math.isfinite(process_peak) or not math.isfinite(free_min) or not 0 < process_peak <= 18432 or free_min < 6144:
        raise RuntimeError("Resident candidate per-card process <=18GiB and actual per-card reserve >=6GiB required")
    return memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="http://10.120.17.101:18005")
    parser.add_argument("--oracle", default="http://10.120.17.101:18004")
    parser.add_argument('--oracle-source-identity', type=Path,
                        default=Path('/data/shaoshitong/search-e1-retriever-segmented-20261007/receipts/source-identity.json'))
    parser.add_argument("--queries", type=Path, required=True,
                        help="Actual validation/training query strings as JSON list or question JSONL")
    parser.add_argument("--memory-evidence", type=Path, required=True,
                        help="Live receipt with actual sample_start/sample_end/updated_at Unix timestamps")
    parser.add_argument("--source-identity", type=Path, required=True,
                        help="Candidate receipts/source-identity.json")
    parser.add_argument("--asset-identity-evidence", type=Path, required=True,
                        help="Owner-verified resident/original full index/corpus/encoder/source SHA comparison receipt")
    parser.add_argument("--cpu-fixture-evidence", type=Path,
                        default=Path(__file__).with_name("cpu-fixture-results.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress-max", action="store_true",
                        help="Candidate only: compare one actual 8-request x 2560-query shared scan")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Gate output already exists; use a new evidence filename')
    token = os.environ["E1_RESIDENT_PROBE_TOKEN"]
    raw = args.queries.read_text()
    try:
        queries = json.loads(raw)
    except ValueError:
        queries = [json.loads(line)["question"] for line in raw.splitlines() if line.strip()]
    if len(queries) < 320 or any(not isinstance(q, str) for q in queries):
        raise ValueError("Use at least 320 actual question strings for the one HTTP integration request")
    gate_start = time.time()
    before = get_json(args.candidate + "/health")
    oracle_health = get_json(args.oracle + "/health")
    oracle_process = verify_oracle(oracle_health, args.oracle_source_identity)
    if not before.get("gpu_ready") or before.get("ntotal") != 21015324 or before.get("d") != 768:
        raise RuntimeError("Candidate is not owner-activated with the full index")
    if not oracle_health.get("ready") or oracle_health.get("ntotal") != 21015324:
        raise RuntimeError("Original full oracle unavailable")
    assets = verified_assets(args.asset_identity_evidence)
    before["gpu_inventory_uuids"] = [g["uuid"] for g in before["gpu_inventory"]]
    if before.get("physical_gpus") != [0,1,2,3] or len(before.get("residency", [])) != 4:
        raise RuntimeError("All four resident shards must be loaded")
    identity = json.loads(args.source_identity.read_text())
    verify_source(identity, before)
    expected_memory_sources=[sha for path,sha in identity['identity']['sources'].items() if Path(path).name=='memory_observer.py']
    if len(expected_memory_sources)!=1:raise RuntimeError('Memory observer source must be captured')
    fixture = json.loads(args.cpu_fixture_evidence.read_text())
    if fixture.get("accepted_cpu_fixture") is not True or fixture.get("full_gpu_accepted") is not False:
        raise RuntimeError("A passed explicitly CPU-only fixture is required")
    scanner_sha = identity["identity"]["original_scan"]["sha256"]
    if scanner_sha != fixture["original_sha256"]:
        raise RuntimeError("CPU fixture used a different original scanner")
    candidate_shas = [sha for source, sha in identity["identity"]["sources"].items() if Path(source).name == "resident_scan.py"]
    if candidate_shas != [fixture["candidate_sha256"]]:
        raise RuntimeError("CPU fixture used a different resident scanner")
    if not any(item["kind"] == "scanner_source" and item["k1_sha256"] == scanner_sha
               for item in assets["comparisons"]):
        raise RuntimeError("Asset proof does not bind the same original scanner")
    encoder_sha = identity["identity"]["original_encoder"]["sha256"]
    if not any(item["kind"] == "encoder_source" and item["k1_sha256"] == encoder_sha
               for item in assets["comparisons"]):
        raise RuntimeError("Asset proof does not bind the same original encoder source")
    sizes = [1, 63, 64, 65, 128, 129, 320, 2560]
    plans = [(sizes, True), (sizes, False)]
    if args.stress_max:
        plans.append(([2560] * 8, True))
    checks = []
    diagnostic = dict(accepted=False, scope="incomplete full GPU and HTTP oracle gate",
                      candidate_pid=before["pid"], source_identity_sha256=before["source_identity_sha256"],
                      candidate_health=before, original_oracle_health=oracle_health, checks=checks,
                      gate_start_unix=gate_start, cpu_fixture=fixture,
                      source_identity=identity, asset_identity_evidence=assets)
    args.output.write_text(json.dumps(diagnostic, indent=2))
    # Exactly one sequential 320-query HTTP integration check on the shared
    # original GPU7 oracle. Never send shape/stress batches or 2560 queries to it.
    integration_payload = dict(queries=queries[:320], topk=3, return_scores=True)
    candidate_started = time.monotonic()
    candidate_integration = post(args.candidate + "/probe/retrieve", integration_payload, token)
    candidate_seconds = time.monotonic() - candidate_started
    original_started = time.monotonic()
    original_integration = post(args.oracle + "/retrieve", integration_payload)
    original_seconds = time.monotonic() - original_started
    if candidate_integration["result"] != original_integration["result"]:
        raise RuntimeError("Single 320-query original HTTP JSON differs, including documents/scores/order")
    diagnostic["gpu7_http_integration"] = dict(queries=320, topk=3, return_scores=True,
        requests_to_k4=1, sequential=True, complete_json_exact=True,
        candidate_seconds=candidate_seconds, oracle_seconds=original_seconds,
        query_sha256=hashlib.sha256(json.dumps(integration_payload["queries"]).encode()).hexdigest())
    args.output.write_text(json.dumps(diagnostic, indent=2))
    for lengths, return_scores in plans:
        payloads = [dict(queries=[queries[(offset * 317 + i) % len(queries)] for i in range(length)],
                         topk=3 + offset % 2, return_scores=return_scores)
                    for offset, length in enumerate(lengths)]
        # The private compare operation is serialized on the single scanner; it
        # never allocates a second scanner or changes production routing.
        candidates = concurrent_requests(args.candidate + "/probe/compare", payloads, token)
        for payload, (candidate, candidate_seconds) in zip(payloads, candidates):
            comparison = candidate["oracle_comparison"]
            if not all(comparison.get(key) is True for key in ["scores_exact", "ids_exact", "shape_exact"]):
                raise RuntimeError("Same-embedding full GPU equality failed")
            checks.append(dict(queries=len(payload["queries"]), topk=payload["topk"],
                               return_scores=return_scores,
                               scope="same-embedding original cuda0 versus resident cuda0..3 full GPU scanner; exact global shapes/merge order",
                               gpu7_http_compared=False, candidate_probe_seconds=candidate_seconds,
                               comparison=comparison,
                               query_sha256=hashlib.sha256(json.dumps(payload["queries"]).encode()).hexdigest()))
        args.output.write_text(json.dumps(diagnostic, indent=2))
        if lengths == [2560] * 8 and any(response["oracle_comparison"]["batch_requests"] != 8
                                      or response["oracle_comparison"]["batch_queries"] != 20480
                                      for response, _ in candidates):
            raise RuntimeError("Stress requests did not meet in one 8-request/20480-query scan; gate remains unaccepted")
    after = get_json(args.candidate + "/health")
    if after["pid"] != before["pid"] or after["source_identity_sha256"] != before["source_identity_sha256"]:
        raise RuntimeError("Candidate identity changed during gate")
    verify_source(json.loads(args.source_identity.read_text()), after)
    verify_oracle(get_json(args.oracle + '/health'), args.oracle_source_identity)
    gate_end = time.time()
    diagnostic["gate_end_unix"] = gate_end
    args.output.write_text(json.dumps(diagnostic, indent=2))
    memory = memory_covering(args.memory_evidence, before, gate_start, gate_end)
    if memory.get('observer',{}).get('source_sha256') != expected_memory_sources[0]:
        raise RuntimeError('Memory observer source is not the reviewed candidate source')
    if not args.stress_max:
        raise RuntimeError("Run --stress-max before production admission; otherwise save diagnostic evidence only")
    receipt = dict(accepted=True, scope="full GPU and HTTP oracle equality",
                   candidate_pid=before["pid"], source_identity_sha256=before["source_identity_sha256"],
                   candidate_health=before, original_oracle_health=oracle_health,
                   cross_device_operator_checks=checks, gpu7_http_integration=diagnostic["gpu7_http_integration"],
                   gate_start_unix=gate_start, gate_end_unix=gate_end, memory=memory,
                   source_identity=identity, asset_identity_evidence=assets, cpu_fixture=fixture,
                   original_oracle_process=oracle_process,
                   evidence_scopes=dict(same_gpu="all listed sizes/topks/tails/stress exact; resident four-device versus original first-device on complete index",
                       gpu7_http="one real 320-query topk3 return_scores=true complete JSON exact; no other sizes",
                       stress="candidate-only 8 x 2560; actual worker requests <=320; not a training budget"),
                   evidence_files_sha256={str(path): file_sha(path) for path in
                       [args.source_identity, args.asset_identity_evidence, args.cpu_fixture_evidence, args.memory_evidence]},
                   note="Probe/stress timings include extra original scans and are not training acceleration evidence")
    args.output.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(accepted=True, checks=len(checks), output=str(args.output))))


if __name__ == "__main__":
    main()
