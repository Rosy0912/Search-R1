#!/usr/bin/env bash
set -euo pipefail
# Candidate only: root must provide actual Base64<26 and new32B hardware receipts.
: "${G1_CANDIDATE_READY_RECEIPT:?Missing independent candidate/native/hardware binding}"
: "${MODEL_PATH:?Own official32B Base fullverified}"
: "${TRAIN_FILE:?Own disjoint train parquet}"
: "${VAL_FILE:?Own disjoint val parquet}"
: "${PROBE_MANIFEST_PATH:?Actual calibrated32B frozen readout manifest}"
: "${G1_REFRESH_CONFIG:?Pinned staged32B calibration configuration}"
export N_GPU=8 PPO_MICRO_BS=64 LOGPROB_MICRO_BS=64 TOTAL_TRAINING_STEPS=101
export ROLLOUT_TP=8 PPO_MINI_BS=256 TRAIN_BS=128
export ADAM_FOREACH=false
export PARAM_OFFLOAD=true GRAD_OFFLOAD=true OPTIM_OFFLOAD=true
# Phase proof must include probe unload during actor update and host envelope.
export G1_CPU_MEMORY_REQUEST_GIB=768
# No launch gate is claimed implemented by a mere nonempty receipt variable.
# This draft deliberately cannot launch until root's full proof gate replaces it.
python -c 'raise SystemExit("CONDITIONAL_CANDIDATE_NO_HARDWARE_PROOF: do not launch")'
