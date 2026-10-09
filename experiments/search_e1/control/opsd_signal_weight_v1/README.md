# Four-card online signal-weight OPSD

This is a new production namespace for the human execution goal dated 2026-10-10.
It does not launch or resume the old archive or the old seven-group study.

Fixed order: G_POS, G_NEG, H_POS, H_NEG. Each receives25 online TRAIN outers,
512 questions x5 fresh student trajectories per outer. Starting source100 and
initial504 FP32 LoRA tensors are common; Adam is independently fresh per arm.

Four physical HKUST4 GPUs4-7, UID1001. Global pair batch32, micro1, accumulation
8 per rank (original eight-card value4), gradient SUM, actual tail denominator.
Eight logical generation workers preserve ranks/seeds42-49 in two four-GPU waves.
No changes to scientific generation budgets, clipping, LR1e-6 or LoRA16/32.

Current-preupdate no_grad scoring uses every sibling's own history and the same B.
Only original uniform selected D trains. G is signed answer-token student-minus-
teacher logprob; H is full-policy full-vocabulary entropy. Center/maxabs, epsilon
1e-6, detached weight1 +/- .9r; group mean1, range.1-1.9. Invalid ancillary encoding
neutralizes the whole question's weights while preserving original D eligibility.

TRAIN EM includes all2560 generation slots, not merely mined/usable pairs, and
belongs to the sampling checkpoint before that outer's updates. Formal native512
runs only once at commonzero and after outer15/25 for each arm (indices14/24 raw
checkpoints). Evaluation uses BF16 source100 plus unmerged FP32 LoRA, original HF
SDPA SearchAgent/greedy/repetition1.05/scorer. Primary endpoint25; no best fallback.

## Deployment and execution

1. Publish this complete helper and a SHA manifest to search-e1; verify FETCH_HEAD
   and actual node files. Do not modify a published/running helper in place.
2. Upload already-published frozen305 science source and its original manifest.
   Copy common initial LoRA HKUST-to-HKUST; large products never land on Mac.
3. Run assets_prepare.py in the original compatible K4 environment with
   CUDA_VISIBLE_DEVICES empty. It reads all12 source files/305 runtime files/data
   and actual504 initial tensors, freezes25 cohorts, and writes remote receipts.
4. Prepare immutable group configs and a study manifest with fresh publication,
   storage/ownership/retriever admission and the exact human-goal binding.
5. Run study_runner.py --study-manifest PATH --study-manifest-sha256 SHA under
   an independent E1_RUN_ID and empty CUDA visibility. The CPU supervisor starts
   each observer and group controller serially. GPU children override visibility.
6. First formal outer/real Adam1/5 must be checked from actual saved states and
   native score/forward receipts; continue the same run, never restart from zero.

Resume is explicit and keeps original run/config/checkpoint/order/RNG. Committed
updates and completed evaluation slots are never regenerated. A failed controller
or incomplete CPU merge/evaluation requires a cause/state audit and independent
recovery source; a missing PID never proves an original return code0.

All weights, complete optimizer/RNG/cursors, cache, raw trajectories, HTTP,
valid-token forward records, score manifests and full evaluation rows stay at
HKUST. Mac retains source/configuration/docs/small summaries only. Node-local
storage is explicitly authorized; independent remote backups are separately
verified and must never be described as complete based on a plan or copy start.

CPU fixtures test equations, state and lifecycle; they do not prove actual GPU
training, source100 EM reproduction, four-rank numerical equality or effects.
The goal is complete only after100 TRAIN rows, all8 formal evaluations, actual
state checks and the paired final analysis have been accepted.
