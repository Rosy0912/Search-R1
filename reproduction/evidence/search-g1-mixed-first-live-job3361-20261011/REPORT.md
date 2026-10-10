# Search-G1 mixed-first + all-correct fallback fresh200

2026-10-11: native job3361 on h100-8; source a76d1a59d53fc286bca4673c0202a6790add1423, based on bf0682ac. Remote base `/mnt/data/shaoshitong/search-g1-mixed-a76d1a5-20261011`.

The 648 deployed files were independently read back with exact SHA matches; 16 CPU tests passed, including actual Hydra micro32/rank4/accum4 composition, source loss contribution equivalence and Adam/scheduler snapshot roundtrip. These tests do not prove production GPU/full-state reload acceptance.

Once-only native submission intent and receipt exist. Job3361 actually progressed from eight-rank mechanics to initial actor/reference/vLLM smoke and then same-cluster local retrieval staging. Source/model/data bindings use official fresh Qwen2.5-3B-Instruct aa8e725 and the canonical manifest; no old model or optimizer was resumed. Existing foreign GPU processes remain untouched under an explicit bound cohost policy.

Completed training outers: 0/200 as of this entry. No reward/EM/F1 result, checkpoint5/50/100/150/200, final51713 result or actual exit0/resource cleanup is claimed yet. Those remain required to complete the goal. Raw trajectories, tensor and checkpoint state stay remote.


## Actual terminal closure

Job3361 failed with NonZeroExitCode1:0 before main_ppo / any Adam write: formal-inline-method.json was absent. Initial model, local retrieval, probe and coexistence checks completed. Owned cleanup completed and original logs retained remotely. No CUDA/OOM cause was observed. Terminal evidence UTC2026-10-10T18:53:10.611667+00:00 is saved separately. No training checkpoint exists; new fresh official initialization remains required.
