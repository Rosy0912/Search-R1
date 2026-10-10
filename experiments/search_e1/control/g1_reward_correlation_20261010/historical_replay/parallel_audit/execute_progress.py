"""Root-invoked published read-only progress request; compact stdout only.

No Node writes/model allocations/signals. Root supplies publication proof whose
FETCH_HEAD file entries match the exact helper/request bytes before execution.
This wrapper stores only compact audit metadata locally.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True); p.add_argument('--publication-proof', required=True)
    p.add_argument('--output-prefix-new', required=True)
    a = p.parse_args(); request_path = Path(a.request).absolute(); reader_path = HERE / 'read_parallel_progress.py'
    raw = request_path.read_bytes(); code = reader_path.read_bytes(); cfg = json.loads(raw)
    proof = json.loads(Path(a.publication_proof).read_bytes())
    assert proof['remote_readback'] == proof['source_commit']
    bysha = {r['sha256'] for r in proof['files']}
    assert hashlib.sha256(raw).hexdigest() in bysha and hashlib.sha256(code).hexdigest() in bysha, 'reader/request bytes not published'
    target = Path(a.output_prefix_new).absolute()
    for suffix in ['.stdout.json', '.stderr.txt', '.execution.json']:
        assert not Path(str(target) + suffix).exists(), 'read observation must use new output prefix'
    sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control'); import control
    cmd = shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
        cfg['original_base'] + '/env/bin/python', '-B', '-c', code.decode()])
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = subprocess.run(control.route('baai-a800', cmd), input=raw, capture_output=True, timeout=180)
    assert len(r.stdout) <= 2 * 1024 * 1024 and len(r.stderr) <= 1024 * 1024, 'unexpected noncompact output retained for root review'
    Path(str(target) + '.stdout.json').write_bytes(r.stdout)
    Path(str(target) + '.stderr.txt').write_bytes(r.stderr)
    Path(str(target) + '.execution.json').write_text(json.dumps(dict(rc=r.returncode, started_utc=started,
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), reader_sha256=hashlib.sha256(code).hexdigest(),
        request_sha256=hashlib.sha256(raw).hexdigest(), source_commit=proof['source_commit'],
        raw_payloads_requested=False, model_or_Torch_imported=False, Node_files_written=False), indent=2) + '\n')
    assert r.returncode == 0, 'read-only audit failed; inspect compact observation without changing running producers'
    print(r.stdout.decode())


if __name__ == '__main__': main()
