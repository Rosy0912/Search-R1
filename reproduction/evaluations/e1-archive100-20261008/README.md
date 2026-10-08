# Search-E1 Archive iter100 fixed512 evaluation source backup

This directory freezes the isolated HKUST5 inference runner and its scoring dependencies. The checkpoint is the Search-E1 Archive four-GPU round-1 GRPO `global_step_100` export. This backup does not execute the runner or contain evaluation results.

## Evaluation protocol

- One GPU: HKUST5 GPU0, pinned by the UUID in `evaluate_worker.py`.
- One greedy response per question; 512 unique questions, NQ and HotpotQA 256 each.
- Maximum five generation turns, four retrieval calls, top-k three passages. Policy spans distinguish model-generated text from retrieved observations.
- EM extracts a legal terminal answer only from the final model-authored span. Crossed/nested actions and observation-only answers receive zero. F1 uses that same extracted answer and maximizes over gold aliases. `answer_format_valid` is terminal answer validity, not whole-trajectory validity.
- All 512 unique question IDs and original question/answer/source fields must match before completion. Duplicate or missing outputs fail coverage checks.

This is fresh greedy inference; it is not an exact replay of the historical 229/512 result. The historical original full512 responses and masks were not persisted. The old Lumia compressed-package score is a separate run.

## External inputs and environment

The 512-question dataset is deliberately excluded. Restore the authorized `data/validation.jsonl` into this directory only from the original verified artifact; required SHA256 is `75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76`. `plan.json` pins the full source and dataset hash manifest. Weights, credentials, environment directories, logs and predictions are excluded.

The runner uses the existing HKUST5 environment `/home/shaoshitong/search-e1-config-study-20261004/env/bin/python`, with PyTorch, Transformers, vLLM, requests and the XFORMERS attention backend. It reads the existing same-cluster retriever at the address in `plan.json`. No dependency installation or model training occurs in these scripts.

`launch_controller.py` expects the adjacent input metadata `../search-e1-archive100-hf-verified-receipt-20261007.json`. It copies checkpoint files from the verified HKUST1 export through an in-memory SSH pipe; complete weights are not saved on the Mac. The destination verifies every SHA256 before loading. The controller uses a task lock and refuses an existing destination; the worker has its own lock and refuses duplicate completion or prediction files.

## Deployment and recovery boundaries

HKUST5 node-local storage was explicitly authorized by the user on October 8, 2026 for this evaluation. The task path is `/home/shaoshitong/search-e1-archive100-eval-20261008`. Scripts never kill existing GPU processes or alter training. GPU availability and UUID are checked immediately before inference. This is an admission check, not a cluster-wide reservation.

Do not rerun a failed worker blindly: checkpoint verification may already have renamed `model.partial` to `model`, and partial predictions must be preserved. Inspect task status, launch receipt, owned processes and file state first. Source checkpoints are retained.

`SOURCE_FILES.json` records the exact original local source locations and hashes. The final publication receipt is kept outside this Git tree to avoid circular commit hashes.

## Optional same-cluster direct transfer

`switch_direct_transfer.py` prepares a task-private HTTPS source service on HKUST1, validates the pinned certificate and an 8 MiB SHA-checked range transfer from HKUST5, and switches only this task's existing Mac pipe if direct transfer is demonstrably faster. It verifies the original controller and SSH child command identities before signaling them, then uses `inspect_target_writers.py` to reject any remaining target task tar/write FD before starting a replacement receiver. It never signals training or unrelated processes.

`direct_https_source.py` exports only the twelve allowlisted immutable checkpoint files. It requires a random Bearer credential, uses TLS 1.2 or newer, checks source file stability, supports SHA-prefix checks and byte ranges, and expires after the configured task lifetime. The receiver trusts the explicit generated task certificate and validates the source IP.

`direct_https_receive.py` preserves existing partial files, verifies source/target prefix SHA before resuming, retains rejected partial copies, and uses at most three concurrent transfers. It flushes and fsyncs each file, reads every full SHA256 back, then registers a single inference worker. It removes only its own random HTTP credential file after transfer.

Generated authentication files, TLS private keys/certificates, runtime status/PID evidence and logs are excluded from this source backup. The 8 MiB route test is only a connectivity/speed sample; completion still requires all twelve file hashes and the inference worker's independent full checkpoint verification.
