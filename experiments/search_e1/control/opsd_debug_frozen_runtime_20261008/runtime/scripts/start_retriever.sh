#!/usr/bin/env bash
# Source from an allocated GPU job. Exact FP32 index spans allocated GPUs; E5 uses the first.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export RETRIEVER_URL=http://127.0.0.1:18000/retrieve
RETRIEVER_BACKEND=${RETRIEVER_BACKEND:-gpu}
if [[ "$RETRIEVER_BACKEND" == cpu ]]; then
  RETRIEVER_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
else
  RETRIEVER_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
CUDA_VISIBLE_DEVICES="$RETRIEVER_DEVICES" "$PYTHON" "$ROOT/scripts/retriever.py" \
 --index /home/jmzhang/self-evolution/Search-R1/data/retriever_data/e5_Flat.index \
 --corpus /home/jmzhang/self-evolution/Search-R1/data/retriever_data/wiki-18.jsonl \
 --model /home/jmzhang/models/e5-base-v2 --backend "$RETRIEVER_BACKEND" \
 --verify-data "$ROOT/data/train.jsonl" --verification-output "$RUN/retrieval_verification.json" > "$RUN/retriever.log" 2>&1 &
RETRIEVER_PID=$!
trap 'kill "$RETRIEVER_PID" 2>/dev/null || true' EXIT
"$PYTHON" - "$RETRIEVER_PID" <<'PY'
import os,sys,time,requests
pid=int(sys.argv[1])
for _ in range(600):
    try:
        os.kill(pid,0)
        r=requests.post('http://127.0.0.1:18000/retrieve',json={'queries':['capital of France'],'topk':3,'return_scores':True},timeout=20)
        r.raise_for_status()
        assert len(r.json()['result'][0]) == 3
        print('Retriever ready: top-3 query passed.',flush=True)
        break
    except ProcessLookupError:
        raise SystemExit('Retriever process exited; inspect retriever.log')
    except (requests.RequestException,AssertionError):
        time.sleep(2)
else:
    raise SystemExit('Retriever startup timed out')
PY
