# v20 XFORMERS正式200执行候选

v20 derives from immutable v19 with sourceebe91, all scientific configuration/data/model/budget/micro/counting/guards and BF16 reduced-precision reduction false unchanged. Only fresh launcher/driver/worker vLLM attention backend is XFORMERS before Ray/Torch and role initialization; CUDA_LAUNCH_BLOCKING=0 restores normal asynchronous execution. Actual all8 runner classes and all36 per-rank XFormersImpl layers are required, using vLLM0.6.3 native sliding_window None. Actor/ref HF attention backend remains unchanged. XFORMERS may change BF16 logits and sampled trajectories; mathematical/scientific settings are retained, bitwise equivalence and GiGPO root-cause repair are not claimed. Own3338 was human-stopped after actual1/20Adam, not a CUDA failure; exact terminal cleanup and no complete checkpoint are bound. Historical1 is never stitched; one fresh official200 starts anew0/0 with initial LR0. Historical512 baseline reuse and nonblocking diagnostics persist; no extra test or initial evaluation job. HUMAN_WAIVED_STATIC_REVIEW is explicit latest human authorization, never a new PASS. Normal Git publication/readback and actual resource/finite/one-submission guards remain mandatory.

Original human review-waiver instruction: 审查啥，你直接跑吧。。别浪费资源，你太慢了

Human stop record: 那你可以直接停掉了，既然100%报错，那不如直接按照目标修复来。。。然后写进本地这个：/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct200/FAILURES-AND-FIXES-20261008.md （增加表格一行）

实际3338人工停止、1outer/20Adam及零残留证据由root提供；本作者未SSH、测试、停止、提交或执行。候选只准备一次新正式作业，不精确恢复、不拼接历史。新源ebe91不改变；44runtime中仅3入口改变，41文件逐字保持。

参考：[官方verl0.4 FAQ](https://verl.readthedocs.io/en/v0.4.0/faq/faq.html#illegal-memory-access)、[vLLM0.6.3 XFormers真实实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.6.3/vllm/attention/backends/xformers.py)。已保存的Search-E1全量生成证据支持规避候选，不能证明GiGPO根因确定或新作业已成功。
