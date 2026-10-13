# Shared launch and predecessor gate review

**ACCEPTED — STATIC_SOURCE_RUNTIME_REVIEW_ACCEPTED. No remaining source blockers.**

Preparation SHA256: `9ca9bb24ea8d8380986252ad26989eb4d148df6aa18e8b891d5060186eb373bb`. Independent no-context review against `b51f1bc895abd520aa201c63bc0aaf388b56dc0c`.

The separate shared-systemd path binds actual MainPID/cgroup, opaque non-Slurm identity, exact eight UUID/OFD leases, numeric vLLM visibility and physical8/reserved0 metadata. It requires68GiB initial GPU free,8GiB continuous headroom,72GiB own cap, actual320GiB host MemAvailable and32 CPU affinity capacity. Load is warning evidence. Original strict Slurm checks remain separate. Both dispatch kinds share the durable once lock/intent namespace; unknown outcomes never automatically retry. Cleanup remains limited to own cgroup/PID identities.

R1 fixed: final launch requires truthful original CANCELLED/0:15, actual replacement COMPLETE51713, full cold HF/tokenizer, separate independent actual acceptance and cost-summary-v2 with underlying full200/replacement/failed-startup/CPU-HF hashes. Source approval or queue absence cannot substitute for final receipts. R2 fixed: prelaunch and early supervisor failures retain actual CPU/wall/error and verified own fd cleanup with honest live/unknown residual; existing body cleanup remains intact.

Core algorithm, canonical config and overrides are byte-for-byte unchanged from the previously reviewed base; no new math retest is claimed. The pinned actual43654-sample report peaks at61776592896 bytes (57.533935546875GiB). Sampling remains a lower bound; admission margin and continuous physical free memory guards remain necessary.

Static parsing, whitespace, exact preparation/core hashes and narrow mocked identity/cost/failure contracts passed. No remote access, GPU calls, model loads, training, commit or push occurred. Actual predecessor acceptance, source publication, live launch progress and new final acceptance remain required.

Exact reviewed file map (all runtime files/docs/config and seven required core files):

| File | SHA256 |
|---|---|
| `reproduction/tree-nested-branch-runtime-20261012/COMPOSITE_PREDECESSOR_CONTRACT.md` | `5a770767980e7e63cca3748fe282a4a7e7e7d10ee9f54c9ad1dc2f98ff41225e` |
| `reproduction/tree-nested-branch-runtime-20261012/LAUNCH_PLAN.md` | `42473604fee20b0526ee7a491a6cee413e465e46bf948cd0caca9932fc0abe4f` |
| `reproduction/tree-nested-branch-runtime-20261012/SHARED_LAUNCH_ADAPTER_PLAN.md` | `42e9994a69ec2392a6ba23009be4aaffda07d9bf0d820516151b99c3fbaada4e` |
| `reproduction/tree-nested-branch-runtime-20261012/attempt_accounting.py` | `00fe13a96e805585278ca1deb9a7bf693fd57923dc7e5c8cf71933f0e489a113` |
| `reproduction/tree-nested-branch-runtime-20261012/build_package.py` | `82767c6f83af84c7a8e24d412671177b100f3c97b6a8958be0b0038d9d02e109` |
| `reproduction/tree-nested-branch-runtime-20261012/build_package_draft.py` | `b07c9cab8c500dcc202a00eda3af724ae14fb27bad1e5a7fa4c9abf0dc27815a` |
| `reproduction/tree-nested-branch-runtime-20261012/calibration_contract.py` | `83de622d96266cf1fa2c141849ae5734aee6eaaf1a9ab84efa86f00ca507f20c` |
| `reproduction/tree-nested-branch-runtime-20261012/config/ppo_tree_nested_branch_fresh200.yaml` | `2e1053b6b1c0b7f86d10c3a3108bc3dbda88e3fce08167eab2b842a8562daab9` |
| `reproduction/tree-nested-branch-runtime-20261012/deploy_package.py` | `e6d23a7a926ddec16ab47f1f0ebaaf87408477e96a366e178a59a9f87cb0f3af` |
| `reproduction/tree-nested-branch-runtime-20261012/expected-gpu-uuids-retained18.json` | `609e1a657db48b39d0548dd61dcbf3bcdfaf1b64d8a968a549f6d3cc71e361db` |
| `reproduction/tree-nested-branch-runtime-20261012/guard_entry.py` | `ef2c38b791b6a48a8f614ae8f4d95b80145e3ae65123b811ea9236af35095099` |
| `reproduction/tree-nested-branch-runtime-20261012/launch.sbatch` | `f11f45682a9da7c483213a7d059867bd5e64f17b66d9b167f763ace6d2afae0a` |
| `reproduction/tree-nested-branch-runtime-20261012/make_release.py` | `505a39dcc7a1ee057f9ebba5e16175a1f6965d640f25334044f0b1428fcd940a` |
| `reproduction/tree-nested-branch-runtime-20261012/manifest.template.json` | `d0fd99da7edab49e16c7e6ae98fd34c2cf0acf2f8d1304d10d9a7c59be31609b` |
| `reproduction/tree-nested-branch-runtime-20261012/owned_lease_identity.py` | `14b3c5ba5fa97a12112f7a929676705c21b0c57105e0b292cdf75b946514bb89` |
| `reproduction/tree-nested-branch-runtime-20261012/partial_raw_evidence.py` | `30138664983d7402aa0b930bdea7f7442960b3c6dd982e0fa7776aadd64fed7e` |
| `reproduction/tree-nested-branch-runtime-20261012/predecessor_gate.py` | `76f596a2ec42603b7b33529d46580a179ed7a057e8da5cc4d7cb8fcb862b532a` |
| `reproduction/tree-nested-branch-runtime-20261012/production_contract.py` | `207dee604802b2fd08b354082dbd598059f393dd576544d3dc638653307852b7` |
| `reproduction/tree-nested-branch-runtime-20261012/render_dispatch.py` | `9716940eb46a8d0d3c380e297ef172aa1097930a79c00bcb34ae17cf8824bee0` |
| `reproduction/tree-nested-branch-runtime-20261012/render_shared_dispatch.py` | `69fa2be0ca3c77bbec7999b9bd52793a3c81fd6890a702eff9a67a7cea3b5790` |
| `reproduction/tree-nested-branch-runtime-20261012/resource_guard_retained18.py` | `4e86d373beb5c62424482efb9f4e6071c5d89d051c9e04b8630db95deed798c5` |
| `reproduction/tree-nested-branch-runtime-20261012/rolling_training_state_hooks.py` | `9540693da4f620bc402fa41c54d13184e09ff871881aaadde10adf1685eb7b38` |
| `reproduction/tree-nested-branch-runtime-20261012/run_tree.sh` | `426541e35cf85e82336acffa9859a8cbc278df962642b00eca189456a73ea3ce` |
| `reproduction/tree-nested-branch-runtime-20261012/shared_guard.py` | `469473b6b7018a3ce24532f33568a310836abc9ac4a1e8063a964a643bc334eb` |
| `reproduction/tree-nested-branch-runtime-20261012/shared_identity.py` | `8f11eb2c42f00fc1ebe3eddf7aafd32f7f0d3aa01f3ea5a3d148e6ed7d0fdf6c` |
| `reproduction/tree-nested-branch-runtime-20261012/shared_launch.py` | `97c4f8e205b4209c3505bdc272eb64572afc66fc740c9a97abbcfe565cc1dda5` |
| `reproduction/tree-nested-branch-runtime-20261012/shared_release_contract.py` | `c47744a4fad2fc59b42a751f9238479836dbcc3c347fece1db6aa2da9f9b727f` |
| `reproduction/tree-nested-branch-runtime-20261012/sitecustomize.py` | `312c652e86f720a0df011d1891295c96c9ada60063d5de2c4981f34a281dcd52` |
| `reproduction/tree-nested-branch-runtime-20261012/slurm_identity.py` | `d438e36f141ae66c8c7a392794f2815d5f90c6b5ad2cc36d730ad72ac0261df6` |
| `reproduction/tree-nested-branch-runtime-20261012/ssh_retained18.py` | `498b50447f37f136ca49cde0cb27ecdcc99db99a425f1b8964de3ac0a231576c` |
| `reproduction/tree-nested-branch-runtime-20261012/storage_guard.py` | `d6bc181faca30f9c9267bb316e4dac2cdfc859e3a320bc66d46d3d3ed3aea425` |
| `reproduction/tree-nested-branch-runtime-20261012/submit_once.py` | `81cbd539b7f0722861d3be89e0dcac620b27b6d9b13dd85514fdafae2d5927ea` |
| `reproduction/tree-nested-branch-runtime-20261012/submit_shared_once.py` | `dadf72328c60c8458410074397d9ce7d7441b1990d48780d28c2cb465ae6ad8f` |
| `reproduction/tree-nested-branch-runtime-20261012/supervisor.py` | `b3fead95b754ddfdfd14ea26c414fa368d72b6f838d553f34a808b75c35b0a37` |
| `reproduction/tree-nested-branch-runtime-20261012/train_monitor.py` | `e6d834d9e61a99d8d378dbb739ef059cf2308c6c142dbeda7bb4a31a4132e2c2` |
| `reproduction/tree-nested-branch-runtime-20261012/training_cost_hooks.py` | `0ad872051b71fd5ce21271e0c05d2412906c3fe4adf5b6413cbc58cf50b0de80` |
| `reproduction/tree-nested-branch-runtime-20261012/validation_cost_entry.py` | `fb993d920d7cab3ef77d63cb4f3eedef7a3e9260b588aec3eb37045567146572` |
| `reproduction/tree-nested-branch-runtime-20261012/validation_cost_hooks_instruct300.py` | `1008f0b9ca472603dafc52f8327c3b871a192cc6cfb4b682ea821ca4c9c823e2` |
| `search_r1/baselines/rollout.py` | `b9e607873cb326a944bdd8a5f56b336c36c9a65d32aac7dc8e0c147287a2049a` |
| `search_r1/baselines/controller.py` | `f78d55d98450614a1a8dac99535277f5cf8a3153f980d8c6f047fea5173b42d1` |
| `search_r1/baselines/tree_audit.py` | `50e7fc4860661b4bd66b7c7acade6c8ec057c499dd6b036ffe0ae240be7fd93d` |
| `search_r1/baselines/tree_fresh200.py` | `9e6c0b7d4af8e2a5a9b072d7c5d2beeda4699f3d1b3360593b0b73367436b7f7` |
| `search_r1/baselines/advantages.py` | `470b84aba5dcb675eb5cadb04e755e4cb6c8a43d916b1bb66d7846cac1a03509` |
| `search_r1/baselines/tree_nested_branch.py` | `34aebb206af4a780ee7a56789df92c8b9a1e3a94995191c7cac2ae868af4f160` |
| `search_r1/baselines/local_runtime.py` | `ff66e807ad15e2ad14cdbab0730f67e2de950cb1b790e1019ac6112ab184901d` |
