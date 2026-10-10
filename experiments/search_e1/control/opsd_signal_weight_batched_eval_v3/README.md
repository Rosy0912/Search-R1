# 后置四卡批量512测试

只准备评测代码，不在训练进行时启动GPU。本入口强制四组100outer训练完成、原续训tmux实际exit0、同一资源锁取得、四张获准卡释放后才可运行。它不能启动训练、重置Adam或停止进程。

复用冻结search_e1.vllm_rollout的执行器、原sampling.merge_latest的CPU PEFT/BF16合并和严格QA评分。固定512题，8个逻辑64题分片、四物理卡分两波、seed42–49，每题1个greedy响应、批量64，5turn/4search/top3/480obs/8192context/每轮500token。SamplingParams明确沿用原greedy的repetition_penalty1.05、EOS151645/151643。XFORMERS/eager/.85不改。每波真实Popen.wait及GPU释放后进入下一波；退出瞬间的只读观察错误不会被当成worker失败或成功。

所有端点使用同一新协议vllm-greedy-rep1p05-bf16-merged-fixed512-v1。重新评共同zero一次，再评G_POS/G_NEG/H_POS/H_NEG的15、25末raw checkpoint，共八次正式测试。HF未合并LoRA与vLLM CPU合并BF16、批量执行和检索请求边界不同，不承诺bitwise、HF239复现或两个协议EM相减；旧HF239只作历史标注。

每个端点保留完整512响应、检索HTTP内容、实际生成ID、严格协议分类和逐题相对新共同baseline的新增/丢失。报告总体与NQ/Hotpot各256EM。产物只写HKUST4 study/batched-endpoint-evaluations，不写Mac。CPU merge不触GPU；父进程不importTorch。主入口必须传已发布publication/pathSHA及原study/pathSHA。仅CPU指标/分片检查不代表GPU评测已通过，运行后才验收512唯一覆盖、八worker真实exit0及统一比较。

运行命令模板：

```sh
CUDA_VISIBLE_DEVICES='' python evaluate.py --mode run \
  --study-manifest /weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/receipts/study-manifest.json \
  --study-manifest-sha256 86db5c8e1a86f6b5fd27f880ca062d193b4623e29869e8fa2aa3dba0e8e46c8e \
  --publication PUBLISHED_RECEIPT_PATH --publication-sha256 PUBLISHED_RECEIPT_SHA
```

已完成正式推理不重跑；如本入口异常，保留其run、partial和真实退出日志，由外部恢复未完成端点/分片，不把缺失结果填成0，也不重启原训练。本初版不声称有自动resume功能。

V2 only replaces the obsolete failed ed7f976 terminal guard with actual port-resume-0146350 owner PID2297236/tick693190113 and original foreground pane2297233 exit0, plus native4/100/100-row terminal ledger. Sampling/merge/scoring/FP32 versus BF16 protocol and all nine evaluation endpoints remain byte-equivalent in behavior. No GPU run until every training outer completes and locks/resources are released. Future replacement supervisor requires another explicit binding, never inferred success.

V3 adds only CPU input export after each full endpoint. Reuses unchanged published build_bank logic, freezes this new vLLM common-zero-success bank BEFORE creating any of the eight new trained endpoint directories, with no extra reference generation. All512 canonical traces remain remote; ordinary entropy later covers no-reference cases and teacher metrics use explicit NA/coverage. The old HF239 bank is historical. No posthoc probabilities are claimed here; a same-checkpoint fixed-HF replay consumer must label retokenized/unprocessed scores and precision differences from merged vLLM generation. No extra pass sampling or training.
