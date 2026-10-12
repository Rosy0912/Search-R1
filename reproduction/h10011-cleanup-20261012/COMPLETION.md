BAPO H100-11 backup and cleanup completed on 2026-10-12 16:47 Asia/Shanghai.

Selected saved outer150 by comparable512 EM236/512 (46.09375%). All40 files (51,934,768,626 bytes) comprising model, full8-rank optimizer/RNG recovery state, algorithm state and necessary logs/config passed official fixed-HF-revision SHA256 and size verification twice. HF revision a463fa2cfd67ece0cdd6a851aff843558832a233, prefix h100-cleanup-20261012/bapo/.

Only exact BAPO obsolete/failed checkpoint, debug, cache and raw training-audit paths were removed after required source and checkpoint backup gates. Failed group reclaimed340,704,337,920 bytes; successful group250,426,535,936 bytes. Total591,130,873,856 bytes. Final /dataset available1,587,544,584,192 bytes. Two separated post-cleanup checks passed deleted-path absence, preserved source, latest200 full state and model, plus unchanged live job allocation. Additional final check confirmed all8 latest200 ranks and3 model shards remain.

Original code checkouts, complete latest200 recovery, TOPD and shared assets were preserved; no training was started, stopped or modified. Other RBA job3387 remained Slurm RUNNING but its log showed repeated activation failures before cleanup; Slurm RUNNING is not proof of normal training. This observation was reported to parent for the separate session; no RBA changes were made.

Source backup includes actual runtime commit0fe642bd,47 current loose runtime/config files,135 failed historical loose files, deployment/recovery/scoring entries and migration/cleanup implementation. See source-receipt.json in local outputs for freshly fetched published SHA and complete tracked-file hashes. Credentials, environment and model weights are excluded from GitHub.
