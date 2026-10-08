# Native prefix fastpath 独立源码审查

结论：最终候选 `8693d0b57406dc87fb96f3aab3538e566c63f621` 相对部署基线 `62271081ba1bb74d8fc3f4958d623758efa74fa5`，未发现需修复的源码问题。审查时间：2026-10-08 06:52:35 UTC。生产文件 SHA256：`8798793a60eb093981b532a490acbab7eafe4688811ab484dcb519d2d63ef384`。

- fastpath 只授权 exact Qwen2TokenizerFast、exact native Tokenizer、同一个 `_tokenizer` backend、cleanup=false，以及受审 HF public/private/converter 函数的 AST 与实际 code/constants/defaults。custom/override/无法验证源码均 fallback。
- `_scan_utf8_prefixes` 仍逐 prefix 调用绑定的 native backend.decode，并与原 bytes 的 UTF8 replacement 结果相等后才记录证明；特殊 token、无效/未完成 UTF8、全部 prefix hash、final bytes 校验均保留。HF `_decode_use_source_tokenizer=False` 副作用也保留。
- runtime 绑定覆盖 public/private decode、code、default/kwdefault 独立快照及 converter；scope 成功退出先 check_runtime，再完整 encoder/file/decoder 复检。审查期间发现的 scope exit/default-cache/converter 初始绑定缺口已在最终候选关闭。
- actual diff 仅生产 `tree_raw_tokens.py` 与两例 CPU fixture。5leaf `[3,2]`、四轮搜索、2048/4096/500/500、采样 IDs/logprobs、PPO 和 μ 路径未改。

审查方式仅本地只读源码/diff/hash，另只读本地 HF 4.44.2 的 decode/_decode/to_py_obj 源。未运行测试、SSH、GPU、job，也未修改或停止部署 3339。作者报告两例真实 HF 小 fixture 通过；最终默认值绑定补充后仅作者报告语法/diff 检查，本审查未独立复跑。此结论只覆盖源码正确性，不能证明真实普通 outer ≤420s，也不构成运行中训练的暂停门禁。
