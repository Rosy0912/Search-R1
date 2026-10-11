User explicitly stopped job3368 to repair performance and subsequently authorized
full recovery from outer5. Old complete6–10 plus partial11 were not checkpointed
and will not be stitched into the new suffix. The old job was observed CANCELLED
0:9; terminal original inode hashes and original9 process/cgroup disappearance
were recorded. A second observation found no native queue or GPU processes;
Slurm had already purged job3368 from its in-memory history. No cleanup receipt
exists, so receipt-based cleanup success is not claimed.

The new native8 profile keeps full CP5 state and the scientific budget, changes
model execution to BF16, uses FlashAttention2 varlen attention for padding
removal and existing sampling journal provenance to omit redundant HF parity
recomputation. It keeps FP32 master weights, Adam, reward/probe and reductions.
This is not bitwise continuation. Historical CP5 validation is reused. New CP10
will be saved before starting11. Actual speed and five resumed outers remain
pending until the new allocation supplies measured evidence.


Published/frozen source653712ba5cb3f7f96dcdf570625cc789b77e3024; full822 Git blobs match independent GitHub readback and target files. Remote focused14CPU tests and3queue tests pass. New3377 submitted once UTC17:31:35; actual UTC17:32 PENDING(Resources), video3371 occupies h100-8. Scheduler estimated Beijing09:15:37 is not an observed start or guarantee. Actual fullGPU restore, varlen model check and resumed6–10/newfullCP10/11start remain pending. User was asked whether to retain exclusive queue or authorize Slurm shared scheduling; absent a new answer, retain3377 and do not alter/cancel the foreign job. No OOM occurred during this recovery submission. Controls here are archival; never replay once-only submission/publication.

## Explicit shared Slurm admission

The human authorized: 允许核实并改为Slurm共用，保留视频和8卡训练预算. Video3371 reserves all eight GPU GRES. Updating pending3377 OverSubscribe to YES succeeded but left it pending Resources. The shared entry requests native96CPU/512GiB with oversubscribe and no additional GPU GRES reservation, then independently verifies all eight actual CUDA UUIDs, bound video job birth and existing process births. Records distinguish reserved GPU count0 from actual shared physical8. Global Slurm/device policy and the video allocation are unchanged. Remote CPU suite23 tests passed; actual GPU restoration and resumed6–10/CP10/11 acceptance remain pending. The old pending3377 may be replaced only while still PENDING; a running healthy job must first be inspected.

UTC18:21:50 ownpending3377 cancelled with RunTime0 and originalcommand checked; video3371 samebirth RUNNING. Once-submitted new3379 using frozen c7785c0 (834 fullGit/targetSHA). UTC18:22 native3379 RUNNING, actual CPU96/mem512G sharing and all8 CUDA UUIDs proved; reservedGPU0 distinct from sharedphysical8. Project GPU restoration and suffixouter completion still pending.

## Removal of extraneous startup numerical gate

User explicitly requested fewer extraneous checks while preserving correct training. Shared3379 FAILED1:0 before newAdam due to the added packed/single BF16 maxlogit threshold. Independent native diagnostic3380 COMPLETED0:0 in11s using savedCP5: different shape max0.8359375, equal-shaped masked reference max0/probability difference0, exact cross-sequence isolation. The numerical gate was a false startup rejection. Removed all diagnostic forwards from startup; retained actual loaded Flash2/dtype configuration and substantive HF/Adam/scheduler/RNG/cursor/probe restoration/finite true-update checks. Original3379 terminal logs/Ray logs fullSHA retained and ownprocess/cgroup cleanup proved; video3371 preserved. No new training suffix counted.
