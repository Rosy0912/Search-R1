"""Owner-run full GPU and HTTP equality gate. Does not activate/admit/switch.

The private compare endpoint compares the full original GPU scanner using the
same encoded query arrays and records actual GEMM shapes. Only one real
320-query request is sent sequentially to the shared HKUST4 HTTP oracle. All
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
            raise RuntimeError("K1/K4 asset/source SHA differs: " + item.get("path", kind))
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
    if memory.get("source_identity_sha256") != before["source_identity_sha256"]:
        raise RuntimeError("Memory evidence must bind the candidate source identity")
    if memory.get("free_measurement") != "nvidia-smi memory.free":
        raise RuntimeError("Memory evidence must query actual memory.free, not compute total minus used")
    process_peak, free_min = float(memory["max_process_mib"]), float(memory["min_free_mib"])
    if not math.isfinite(process_peak) or not math.isfinite(free_min) or not 0 < process_peak <= 1024 or free_min < 1024:
        raise RuntimeError("Actual candidate process <= 1 GiB and whole GPU reserve >= 1 GiB required")
    return memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="http://10.120.17.101:18004")
    parser.add_argument("--oracle", default="http://10.123.4.11:18003")
    parser.add_argument("--queries", type=Path, required=True,
                        help="Actual validation/training query strings as JSON list or question JSONL")
    parser.add_argument("--memory-evidence", type=Path, required=True,
                        help="Live receipt with actual sample_start/sample_end/updated_at Unix timestamps")
    parser.add_argument("--source-identity", type=Path, required=True,
                        help="Candidate receipts/source-identity.json")
    parser.add_argument("--asset-identity-evidence", type=Path, required=True,
                        help="Owner-verified K1/K4 full index/corpus/encoder/source SHA comparison receipt")
    parser.add_argument("--cpu-fixture-evidence", type=Path,
                        default=Path(__file__).with_name("cpu-fixture-results.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress-max", action="store_true",
                        help="Candidate only: compare one actual 8-request x 2560-query shared scan")
    args = parser.parse_args()
    token = os.environ["E1_SEGMENTED_PROBE_TOKEN"]
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
    if not before.get("gpu_ready") or before.get("ntotal") != 21015324 or before.get("d") != 768:
        raise RuntimeError("Candidate is not owner-activated with the full index")
    if not oracle_health.get("ready") or oracle_health.get("ntotal") != 21015324:
        raise RuntimeError("Original full oracle unavailable")
    assets = verified_assets(args.asset_identity_evidence)
    identity = json.loads(args.source_identity.read_text())
    if identity["pid"] != before["pid"] or identity["source_identity_sha256"] != before["source_identity_sha256"]:
        raise RuntimeError("Candidate source identity receipt does not match the live process")
    fixture = json.loads(args.cpu_fixture_evidence.read_text())
    if fixture.get("accepted_cpu_fixture") is not True or fixture.get("full_gpu_accepted") is not False:
        raise RuntimeError("A passed explicitly CPU-only fixture is required")
    scanner_sha = identity["identity"]["original_scan"]["sha256"]
    if scanner_sha != fixture["original_sha256"]:
        raise RuntimeError("CPU fixture used a different original scanner")
    candidate_shas = [sha for source, sha in identity["identity"]["sources"].items() if Path(source).name == "segmented_scan.py"]
    if candidate_shas != [fixture["candidate_sha256"]]:
        raise RuntimeError("CPU fixture used a different segmented scanner")
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
    # HKUST4 oracle. Never send shape/stress batches or 2560 queries to it.
    integration_payload = dict(queries=queries[:320], topk=3, return_scores=True)
    candidate_started = time.monotonic()
    candidate_integration = post(args.candidate + "/probe/retrieve", integration_payload, token)
    candidate_seconds = time.monotonic() - candidate_started
    original_started = time.monotonic()
    original_integration = post(args.oracle + "/retrieve", integration_payload)
    original_seconds = time.monotonic() - original_started
    if candidate_integration["result"] != original_integration["result"]:
        raise RuntimeError("Single 320-query original HTTP JSON differs, including documents/scores/order")
    diagnostic["k4_http_integration"] = dict(queries=320, topk=3, return_scores=True,
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
                               scope="same-embedding original versus segmented full GPU scanner on candidate only",
                               k4_http_compared=False, candidate_probe_seconds=candidate_seconds,
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
    gate_end = time.time()
    diagnostic["gate_end_unix"] = gate_end
    args.output.write_text(json.dumps(diagnostic, indent=2))
    memory = memory_covering(args.memory_evidence, before, gate_start, gate_end)
    if not args.stress_max:
        raise RuntimeError("Run --stress-max before production admission; otherwise save diagnostic evidence only")
    receipt = dict(accepted=True, scope="full GPU and HTTP oracle equality",
                   candidate_pid=before["pid"], source_identity_sha256=before["source_identity_sha256"],
                   candidate_health=before, original_oracle_health=oracle_health,
                   same_gpu_operator_checks=checks, k4_http_integration=diagnostic["k4_http_integration"],
                   gate_start_unix=gate_start, gate_end_unix=gate_end, memory=memory,
                   source_identity=identity, asset_identity_evidence=assets, cpu_fixture=fixture,
                   evidence_scopes=dict(same_gpu="all listed sizes/topks/tails/stress exact on complete index",
                       k4_http="one real 320-query topk3 return_scores=true complete JSON exact; no other sizes",
                       stress="candidate-only 8 x 2560; actual worker requests <=320; not a training budget"),
                   evidence_files_sha256={str(path): file_sha(path) for path in
                       [args.source_identity, args.asset_identity_evidence, args.cpu_fixture_evidence, args.memory_evidence]},
                   note="Probe/stress timings include extra original scans and are not training acceleration evidence")
    args.output.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(accepted=True, checks=len(checks), output=str(args.output))))


if __name__ == "__main__":
    main()
