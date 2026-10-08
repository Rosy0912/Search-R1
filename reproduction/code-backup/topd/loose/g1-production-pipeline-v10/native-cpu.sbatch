#!/bin/bash
#SBATCH --job-name=g1-pipeline-cpu
#SBATCH --partition=gpu
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/g1-pipeline-profile-cpu-%j.log
#SBATCH --nodelist=h100-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
: "${PIPELINE_ROOT:?frozen pipeline directory}"
: "${CANDIDATE_ROOT:?frozen v3 source directory}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TASK_ROOT/test-tools:$PIPELINE_ROOT:$CANDIDATE_ROOT"
OUT="$TASK_ROOT/logs/g1-pipeline-native-cpu-$SLURM_JOB_ID"
mkdir -p "$OUT"
 : "${PIPELINE_MANIFEST_SHA256:?exact frozen manifestSHA}"
"$TASK_ROOT/env/bin/python" - "$PIPELINE_ROOT" "$PIPELINE_MANIFEST_SHA256" "$CANDIDATE_ROOT" <<'VERIFY_SOURCE'
import pathlib,json,sys,hashlib
sys.path.insert(0,sys.argv[1]);from pipeline import verify_files
for directory,relative,expected in ((sys.argv[1],'files.json',sys.argv[2]),(sys.argv[3],'reproduction/conditional-teacher100/files.json','8f6582c4cb31ccbdbef62cfbe602981d1cf7032b69a01655604908d64fb783ff')):
 root=pathlib.Path(directory);p=root/relative
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 records=dict(json.loads(p.read_text())['files']);records[relative]={'size':p.stat().st_size,'sha256':expected};verify_files(root,records)
 print('EXACT_SOURCE_MANIFEST',str(p),expected)
VERIFY_SOURCE
"$TASK_ROOT/env/bin/python" -m pytest -q "$PIPELINE_ROOT/test_pipeline.py" "$PIPELINE_ROOT/test_guardian.py" "$PIPELINE_ROOT/test_retrieval_gate.py" "$PIPELINE_ROOT/test_retrieval_startup.py" "$PIPELINE_ROOT/test_resource_audit.py" "$PIPELINE_ROOT/test_prepare_run.py" "$PIPELINE_ROOT/test_terminal.py" "$PIPELINE_ROOT/test_final_cleanup.py" "$PIPELINE_ROOT/test_health.py" "$PIPELINE_ROOT/test_rpc.py" "$PIPELINE_ROOT/test_controlled_exit.py" "$PIPELINE_ROOT/test_retrieval_http_audit.py" "$CANDIDATE_ROOT/reproduction/conditional-teacher100/test_conditional_candidate.py" --basetemp "$OUT/pytest-temp" -o "cache_dir=$OUT/pytest-cache"
