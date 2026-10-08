# Independent OPSD epoch1 selection and merged-weight audit

Actual CPU audit: 2026-10-08T10:16:59.437219+08:00 to 2026-10-08T10:17:39.690934+08:00. Run 914bfff786c84ede8d82414dff65f60e on HKUST1 /data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007. All inspection was read-only; no GPU context, inference/retrieval workload, code/config/runtime/route/affinity change, process signal or publisher edit was performed.

The native512 history is baseline239/512=46.6796875%, raw OPSD epoch1 203/512=39.6484375%, a loss of36 answers or7.03125 percentage points. Raw training loss is0.17748732205277348. The same actual512 file (NQ256 and Hotpot256) and the same native greedy SearchAgent/global reduction function were used before and after training, split into64 rows per each of eight ranks. Model weights changed. This is an aggregate decline for this run/configuration; no per-question paired outcomes or statistical significance are inferred.

Actual adapter_epoch_1 has504 finite tensors, all504 changed from the baseline. Its SHA256 is 8eef75aafa06eb98dc21a6b0ab85195800fc373692d633faa0ecf461fa45afd0. The selected best_adapter has504 finite tensors and252 LoRA-B tensors all exactly zero. SHA256 8cbebc6cb10203bf93a2c8b7271d31020ab1d1e7f1958a6f94f5890af3206c3d is byte-identical to the actual initial baseline file recorded during the independent five-update audit. Thus the learned raw checkpoint exists and is finite; the selected checkpoint has reverted to the baseline through the original selection policy.

Frozen opsd.py (SHA256 61df9cc630bca7466f3edfb06da18ad92ddff1c713bf7c6c4a104ba88672c037) writes the initial baseline as best_adapter, overwrites it only when epoch EM is strictly better, then explicitly reloads best_adapter before merge. Because203 is less than239, epoch1 is not selected.

Complete actual tensor comparison additionally proves the merged model is the source100 baseline at its BF16 loading precision: 13588414464 source FP32 payload bytes were read on CPU; all435 source tensors, cast exactly as the native program loads them, equal their434 merged tensor targets. The sole missing serialized lm_head.weight is tied to model.embed_tokens.weight and was also compared explicitly. Every compared tensor is finite, with no value mismatch, and all source/merged file metadata stayed unchanged throughout. The merged shard layout and dtype differ from the source FP32 files, so this is full normalized tensor equivalence, not whole-file byte equivalence or a fresh raw-file SHA claim. Tokenizer/generation files also match source bytes.

Current final-test parent3960546 and eight owned UID1005 workers3960643/3960647-3960653 read runs/opsd/merged with k1 and temperature0, using the unchanged51713-row seven-benchmark test data. Therefore the current seven-benchmark job evaluates the selected baseline; its result must not be presented as the raw trained epoch1 model's performance or a positive OPSD training gain. Keep the baseline, raw epoch1 and selected-policy result separate. Raw adapter_epoch_1 remains available for later matched evaluation.

The immutable raw audit retains a harmless checker-string finding: parent argv has temperature string0, child argparse serialization has string0.0. A separate interpretation verifies all nine numeric values equal zero plus exact eight GPU ownership/model/run/k1 fields; nothing about the actual run was changed.

Evidence:

- History SHA256 44b2776b9a278518dc9c459cae0f224e5ed1a5cd47f8da1fc9ace05d836d7aa9; actual512 data SHA256 75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76.
- Raw CPU tensor/source/process receipt: preparation-3-epoch1-selection-and-merged-weight-readonly-1791425690.json, SHA256 5f618b304b05d780c8c6ddd87d67a23f3bf5ea27c8965646adc67a14b2dd23d9.
- Numeric-argv interpretation: preparation-3-epoch1-selection-and-merged-weight-readonly-1791425690-numeric-argv-interpretation.json, SHA256 36c7e1638c45f7653969cf164a1a54a7198a6fc016fd796315885cd596da1116.
- Merged config SHA256 44edc33861769e737120b9d0c3a24dfa0c695b92cc149c16d94219c92faf719d, model_type qwen2, dtype bfloat16, source_name_or_path assets/archive100.

Five real updates previously passed. This audit confirms the subsequent native512 degradation and selection fallback; it does not claim that complete seven-benchmark evaluation or effect attribution is finished.
