#!/bin/bash
set -euo pipefail
ROOT=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009
RUN=${BAPO_RUN_ID:?}
cd "$ROOT/repo"
export NUMERIC_FIXTURE_OUTPUT="$RUN/numeric-fixture"
port=$("$ROOT/env/bin/python" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
"$ROOT/env/bin/python" -B -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/tools/deployment/numerics_gpu_fixture.py" > "$RUN/numeric-fixture.log" 2>&1
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/capture_retrieval_golden.py" --output "$RUN/retrieval-golden.json" > "$RUN/retrieval-golden.log" 2>&1
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/bind_retrieval_proof.py" capture
# Capture exits before the live retriever starts, so full GPU indices do not coexist.
"$ROOT/env/bin/python" -B -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",18083));s.close()'
export SEARCH_R1_RETRIEVER_IDENTITY=1
"$ROOT/env/bin/python" -B reproduction/h20/retriever_local.py --index_path "$ROOT/assets-root/assets/wiki/e5_Flat.index" --corpus_path "$ROOT/assets-root/assets/wiki/wiki-18.jsonl" --retriever_model "$ROOT/assets-root/assets/e5-base-v2" --device cuda --faiss_gpu --host 127.0.0.1 --port 18083 --threads 8 > "$RUN/retriever.log" 2>&1 &
retriever_pid=$!
export BAPO_LOCAL_RETRIEVER_PID="$retriever_pid" BAPO_RETRIEVER_PARENT_PID="$$"
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/bind_retrieval_proof.py" process
ready=false
for ((i=0;i<360;i++)); do
 kill -0 "$retriever_pid"
 if curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:18083/identity > "$RUN/identity-ready.tmp"; then ready=true;break;fi
 sleep 5
done
[[ "$ready" = true ]]
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/verify_live_retrieval.py"
