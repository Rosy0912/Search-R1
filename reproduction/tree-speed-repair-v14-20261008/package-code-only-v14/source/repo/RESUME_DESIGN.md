# Tree checkpoint100 recovery to300

Authorization: current user goal explicitly requests recovery from the best recent checkpoint, larger batches, exactly ten optimizer calls per outer, continuation to300, reward/development-collapse handling, and complete multi-rollout search trajectories. This supersedes the earlier fixed512/mini128 contract for the recovery run only. Preserve original3154 evidence.

## Actual evidence

3154 was still Slurm RUNNING at262, but its behavior collapsed. The first zero training reward is138 (later tiny nonzero batches do not establish recovery). Fixed512 dev at150/200/250 has F1=EM=0. At136 mean preclip gradient norm=2.9125918963322294e14, PG loss=4.843696779913333e10. No nonfinite optimizer exception was required for behavior collapse. Best *evaluated* checkpoint is100: F1 .48918324738300495, EM203/512;50 F1 .4645926339285715, EM202/512.125 exists but has no fixed evaluation; it cannot be called better from training reward alone.

Old token PPO exp(current-old) has an unbounded negative-advantage tail. Repeated updates against the same old logprob can amplify that tail. This is a supported mechanism and investigation target, not yet a proven cause of the observed collapse; historic individual ratio tensors were not retained. Do not silently clamp ratios/change the algorithm and call it the same objective.

## Recovery contract

Use checkpoint100 on h100-9, preserve official base reference separately. Start labeled logical outer101, finish300, distinguish original abandoned updates101..262 from accepted recovery progress. Original actor/HF, full8rank Adam/scheduler and driver files must be verified before loading. Old restore function is same-live-only and rejects empty initialized Adam; cold restart needs explicit layout/identity/hash validation and tested state migration. Batch change invalidates exact executor/dataloader equivalence claims. Preserve sampler order/cursor and Tree RNG with a documented batch-size migration; keep original identities as parent provenance and new config/source/job identities as recovery identities. If full state cannot safely migrate, do not quietly fall back to reset Adam; record and resolve the limitation.

Use train640 questions, six leaves3840, global mini384, global actor micro16 (per rank2), epochs1: ten optimizer calls/rank per outer, 48local minirows/rank. Preserve old/reference scoring global32 and fixed dev512 batch32; batch640 fits169028rows. Maintain lengths4096/2048/500/500, four searches, Tree2roots×3leaves, KL.001, LR1e-6 with prior57warmup completed; scheduler advances once per successful outer. Larger effective mini through accumulation avoids increasing already high instantaneous GPU memory. Detect and test final partial batch behavior; no padded question duplicates in objective.

Every search rollout must persist complete question/trajectory/tree IDs, full prompt/action/tool observations and raw token IDs, action records/turn IDs, masks, reward and all leaves; preserve tensors old/ref logprob, advantages and weights before actor update. Atomic exclusive files plus SHA manifests; diagnostic failure must be visible. Save ratio-tail and component-loss statistics for every optimization call, and failing/spiking accumulated tensors before overwriting them. Preserve logs for failed/recovered attempts separately.

Evaluate restored checkpoint100 before any optimizer update against512 baseline; require no collapse and compare actual score, not only file existence. Check every successful outer actual ten calls/all8rank, finite norms/parameters and real nonzero updates. Record training reward, action/search/finish rates; fail closed on clearly collapsed behavior (persist evidence first) and evaluate periodically sufficiently often to catch degradation. Collapse thresholds and event handling must be explicit, not fabricated after observing results. Recovery checkpoint choice stays fixed-dev-based; no final-test selection. Restore from last accepted healthy checkpoint on a proven collapse with preserved failed-attempt trajectories and diagnosis.

Final completion remains logical300 plus seven datasets51713, actual checkpoint/tokenizer and fullstate reload, independent tree_review acceptance, original8background PIDs preserved, own cleanup, separate original/recovery/validation costs and changed sampling/optimization budgets. Same-node GPU E5,8GPU FP16 FAISS, local assets/Ray/HTTP, leases/dual15%foreign thresholds unchanged. No peer task restarts and no new cross-cluster transfers.

## Implementation and acceptance sequence

1. Preserve source logs/cost/Slurm and cancel only the collapsed3154; verify terminal cleanup and background processes.
2. Independent review of selection/causal scope and recovery contract.
3. Verify100 full serialized files/HF and build a separate frozen recovery candidate; unit-test cold optimizer/layout loading, sampler migration, ten-step accumulation and full trajectory publication with failure preservation.
4. Independently review source/config and dispatch once into the same h100-9 resource guard protocol. Actual restored512 evaluation and initial recovery updates are acceptance evidence, never simulated as successful training.
5. Continue101..300 with trajectory evidence and collapse recovery, then full final acceptance. No completion claim until actual outcomes.

Recovery quality evaluation consumes its normal driver/worker generation state. The implementation does not rewind that evaluation or pretend a fresh vLLM engine has the original executor RNG. Parent Tree RNG and sampler resume at their saved boundary; subsequent initial-dev consumption is explicitly part of this recovery protocol. Native cold replay verifies only its scoped synthetic FSDP/Adam case, followed by actual3B state loading, fixed512 score, and five real recovery updates before106 release.
