# C1 + P + R online OPSD, 100 rounds

The latest human goal authorizes HKUST1's existing eight GPUs for a fresh run from the common Search-R1/archive GRPO checkpoint100. The first ten training rounds and first complete native512 evaluation are the operating acceptance milestone; the same automatic pipeline then continues to100 rounds and ten512 evaluations.

| Setting | Value |
| --- | --- |
| Training start | Accepted source100 weights and original common504 FP32 initial LoRA; fresh Adam |
| C1 science | LR1e-6 constant, clip10, LoRA16/32, trajectory-internal valid-token mean followed by actual global-pair mean, global32 |
| P precision | FP32 master/gradients/Adam; FP16 major computation and sampling, FP32 stability exceptions |
| R student selection | Uniform among all five fresh trajectories, including the selected successful reference |
| Teacher / sample weight | Fixed privileged source100 teacher; unit weight |
| Sampling |100 online rounds,512 TRAIN questions ×5 trajectories per round, original seed/budgets |
| Evaluation | Raw current checkpoint after10,20,...100; original ordinary greedy BF16base + FP32adapter native512 protocol |
| Continuation | Same immutable100-round config/run and full Adam/RNG/cursor state across evaluation boundaries |

New output root: `/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010`. Model, data, checkpoint, cache, generation and evaluation files remain on HKUST. Mac stores source, configuration and small status/report aggregates only. Existing eight-card work is allowed to finish and release before the new run starts; no other task is stopped.

The latest human output policy supersedes the older exhaustive dump policy. Top32 and full-vocabulary NPY snapshots and autonomous teacher diagnostic generation are disabled. Necessary training/mining inputs, retrieval/generation logs and compact valid-token losses remain available for continuation and analysis. Protect initial/update1/update5, full outer-end states and the latest two full states; only other new-run intermediate checkpoint payloads are reclaimed with preserved manifests/receipts. Completed new-run merged-model caches are reclaimed after the real trainer exits0 and its GPU children release. Old experiments and active state are excluded.

HKUST1 mount inspection found `/data` on ext4 `/dev/sdb1` with approximately752GiB available and the root volume with230GiB. This reduced policy fits without deleting unrelated account assets. If space later becomes insufficient, use an available remote mount or the latest authorized account cleanup scope, checking active dependencies first. No new Mac mount stores experiment payloads.

Implementation namespaces are `opsd_pr100_training_v1`, `opsd_pr100_periodic_v1` and `opsd_pr100_launch_v1`. Training/evaluation share the existing GPU leases; the CPU supervisor holds a separate pipeline lock and records each actual child return code before advancing. Backup does not gate GPU startup. The baseline239/512 is the already measured identical source/runtime/data/protocol BF16zero, not a copied historical claim. Periodic outputs record strict EM, NQ/Hotpot and same-question gained/lost counts. TRAIN sampling EM and loss are separate from heldout512 EM.

The old eight-card evaluation workers completed and released. The old parent subsequently returned2 in cross-group summary collection; preserve that actual failure rather than requiring or fabricating0. New-goal resource transition binds the exact observed terminal and fresh free/oracle/ownership checks, so the old CPU summary does not delay the new GPU job.

Actual launch: deployed source `dc08aad44fe42adbc24ce102a774735a2c66f4f8`,42 payload files read back from FETCH_HEAD and HKUST1. Training run `c6a79477161a4a059e4fd68fc7567f87`, config `receipts/training-config.json` SHA `651dfb40b1bf22fb5f604b9b17c45ffe8619f20baf521f399bc8059b11d6aa91`. Pipeline run `01e023be640c437aadea6a97da5819c6`, config `periodic/receipts/pipeline-config.json` SHA `3a7643138883c63287db1517f574fe4642395da4bf164df238d9b7b5412435d6`. Tmux session is `e1-pr100-20261010`; pipeline PID1146480/UID1005/starttick935754892 and training PID1146918/starttick935755436 were independently live during first-outer sampling. Observer PID1146725/starttick935755068 records direct eight-card free memory. This proves actual launch and sampling, with zero committed Adam updates at that snapshot; it is not first10 or performance acceptance.

The acceptance evidence is actual training state and `periodic/pipeline/first10-training-and512-accepted.json`, not background launch, CPU tests or model initialization. The final100-round outcome is `periodic/pipeline/pipeline-complete.json` plus ten complete512 summaries. Necessary delta checks were performed without new reviewer rounds or repeated full audits.
