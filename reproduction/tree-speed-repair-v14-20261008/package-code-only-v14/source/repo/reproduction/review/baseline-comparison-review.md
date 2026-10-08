# Search-R1 与所给基线的追加运行审查

日期：2026-10-02。比较对象为当前 `Documents/Search-R1/repo` 与提供的 `Documents/Search-G1/repo`。后者README第1行明确为 **Search-G1**，不是Search-E1；在用户澄清之前，本报告不将其称作E1或推断E1算法。初次检查只读，没有Git操作、停止进程或GPU计算；随后应root明确授权，仅修改本地两份只读checkpoint验收器及其CPU测试。

**结论：没有发现需要中止当前500步训练的新算法错误。基线不能直接移植；发现一个checkpoint完整性验收缺口，已本地补齐，待root目标CPU测试及部署。另需留存本轮实际运行版本证据。** 当前运行状态是root报告的step2完成、step3进行，不是五步或最终论文验收。

## 基线身份与不可直接比较的变量

| 变量 | 当前Search-R1 | 所给Search-G1 | 审查含义 |
|---|---|---|---|
| 底座 | `reproduction/h20/train_paper_grpo.sh:15`使用本地Qwen2.5-3B **base** | `Search-G1/README.md:56`及179行示例为Qwen2.5-3B-**Instruct** | 格式遵循、初始EM及早期reward不可直接对比；不能把换Instruct叫作base配方修复。 |
| 训练入口与奖励 | `train_paper_grpo.sh:20`调用main_ppo；`verl/trainer/main_ppo.py:82–98`只对模型生成span计算EM，错误或无完整答案为0 | `Search-G1/recipes/train_search_g1.sh:26–35`调用main_ppo_eco并开启probe；`verl/trainer/main_ppo_eco.py:271–313`有grounding bonus、wrong-answer floor与重复检索cost | G1 reward不是纯EM，可超过1，错误也可有正floor。曲线上升不证明R1实现错误，也不支持把G1 shaping、probe、cost或wrong_floor偷偷加进纯EM实验。 |
| 预算与batch | R1 `train_paper_grpo.sh:26,40,63,71`：batch512、mini256、8GPU、上限501实际500更新 | G1 recipe 11–19、40–57、82行：默认4GPU、batch128、mini32、上限250 | 同step数和reward不等于同样本/rollout/优化预算。若另开G1对照须独立实验及预算表，不能热改当前R1。 |
| 评估 | R1默认64题、终端全量7源，drop_last=False | G1默认val_data_num200、batch64；`ray_trainer.py:410–414`为drop_last=True | G1会丢不整除的评测尾部，不能照搬。R1已修尾部计数，最终必须与全量audit逐源一致。 |

G1生成padding分支 `search_r1/llm_agent/generation.py:197`重新构造DataProto不传meta_info；R1在198–200行显式保留meta_info，以保持验证do_sample/temperature等设置。整段拷贝G1运行代码会重新引入已修的尾部及生成metadata问题。G1的多指标/成本分析可作为以后独立分析工具参考，不是当前EM训练所必需的依赖。

## 当前适配核查

- 64题在 `prepare_paper_data.py:13–27`按7源、各4个长度层、固定SHA256顺序抽取，NQ10题/其余各9题，检查index及问题唯一。它是早期诊断集，单源9–10题的EM噪声较大；不作为论文性能最终数字。遵守用户32–64题要求，不建议为早期验收扩大回全量。
- `train_paper_grpo.sh:22–23,66,71`分别设置64题val、全量final_val、额外step5 save和501计数上限。`ray_trainer.py:872–880`只在终端切换全量；第五步保存不会return或重置，保持同进程500更新。首步warmup学习率0意味着“5次真实iteration”不能表述为每步非零权重变化。
- GPU检索使用原检索实现的 `co.useFloat16=True/co.shard=True`（`retrieval_server.py:213–215`）。隔离overlay只作用检索子进程，训练环境未换FAISS/NumPy。`gpu_retrieval_preflight.py:28–40`的14题41/42 top3重合是小样本，不等于全体等价；512计时由tile展开向量，仅测search，不含编码/HTTP。
- `retrieval_server.py:327–347`改async端点且body同步、无await/offload，串行在资源初始化的调用线程；排序/JSON协议未变。root报告HTTP及64题真实验证通过。不能把Torch-first import预载当作已证实的cuBLAS根因：root已确认两种import顺序同用项目cuBLAS12.4。
- `run_training.sh:11–25,34–56`保留锁、已成功前置证据、GPUguard、pidfd-owned retriever和真实HTTP门禁；只读五步观察器不会暂停500步trainer。

## Confirmed finding：checkpoint自洽不等于模型完整（已本地修复）

修复前 `verify_five_updates.py:33–38` 与 `verify_full_run.py:9–15`仅将每片header与checkpoint自己index的keys对比。因此，删掉一个参数和相应index entry，同时保留自洽的其它分片，仍会通过“完整”门禁；五步只检查q_proj有限，非抽查参数的NaN也不会发现，full没有tensor有限性检查。这是**验收假阳性**，没有证据表示实际训练checkpoint已经损坏。

按root授权，当前两份本地canonical verifier已各自补充独立函数：

1. 将saved keys与经官方资源校验的底座**实际序列化keys**精确对照；不用模型模块名猜测tied-weight key。
2. 逐参数对照底座header shape，保留每片header/index一致性；文件名和分片划分可不同，合法reshard不拒绝。
3. 分片及index均须本轮fresh；逐参数读取CPU tensor检查有限性后立即释放，不在GPU运行或一次常驻整模型。
4. 五步仍要求选定参数真实变化、steps1–5真实metrics；final仍要求7源完整计数及EM范围。

对应当前行证据：`verify_five_updates.py:33–56`、`verify_full_run.py:10–33`。新增8项真实CPU safetensors回归，两个gate分别拒绝自洽缺key、wrong shape、非抽查参数NaN，并接受合法reshard（`tests/reproduction/test_acceptance_gates.py`）。本地3文件AST语法通过；本机缺safetensors，root随后在目标实际环境统一执行：**55项CPU测试全部通过、0skip**（0.258秒），新增验收器已在第五步前作为只读脚本部署。本reviewer未执行GPU或更改训练行为、活动远端文件或Git。

## 必要结果留档

`run_training.sh:8,53`记录本轮开始时刻，但没有将本轮实际训练源码commit/diff/hash、完整resolved Hydra配置和基线身份组成独立run manifest。审查时当前仓库存在未提交改动；bootstrap更早的environment-source-commit不能自动等同于当前trainer已加载版本。发布最终指标前，应保留现有初始化config日志，并据部署/启动记录留存本轮训练源码快照、底座revision、数据/索引hash、检索wheel/模式、64题sample ids和验收脚本版本。事后HEAD只能证明记录时版本，不能无说明冒称进程启动时版本；此项不需要停止正在训练的进程。

root已补 `reproduction/evidence/active-run-source-snapshot.json`，记录6个核心/launcher远端磁盘hash与本地一致，明确这是采集时的on-disk快照、不是loaded Python module证明。这一透明边界处理合理；后续仍应保存已打印的resolved配置和部署时间线。root最后报告step4完成（reward mean0.041）、step5进行，**尚未五步验收通过**。

不要因G1 reward看起来更高而更换底座、修改EM、打开强制回答、改组大小或引入probe。若用户最终确认要比较的是Search-E1，应先读取正确仓库，再单独审查其算法和配方，不借G1名称补推断。完整Search-R1结果仍须实际五步验收与最终500步checkpoint/全量评估证据。

## 实际第五步后的序列化规则修正

实际第五步完成后，新增严格gate首次拒绝checkpoint，原因是**本审查补丁最初对canonical serialized keys作了过严的exact-set要求**，没有覆盖FSDP/HF保存时的合法tied-weight冗余。这不是训练缺少参数的证据。本reviewer随后只读远端确认：底座434 keys、checkpoint435 keys，canonical keys无缺失，唯一extra为 `lm_head.weight`；两份真实config均为Qwen2ForCausalLM且tie_word_embeddings=True。root CPU检查saved embed/head的151936×2048 FP32 tensor精确相等、均有限。

保存链为 `fsdp_workers.py:509–527`：FSDP FULL_STATE_DICT→HF save_pretrained。[Transformers4.46.3 Qwen2源码](https://raw.githubusercontent.com/huggingface/transformers/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py)声明 `_tied_weights_keys=['lm_head.weight']`；canonical发布文件的去冗余形式与本次FSDP收集后保存的重复形式可以不同。

应root明确授权，两gate本地修正为**当前Qwen profile的窄规则**：所有canonical keys仍必须存在；仅可额外包含lm_head.weight，且canonical有embed/无head，两份config均严格匹配model_type=qwen2、architectures=[Qwen2ForCausalLM]、tie_word_embeddings=true。额外head shape必须对应canonical embed；所有saved tensor仍逐CPU检查finite；saved embed/head还必须dtype相同、torch.equal精确相等。任意其它extra、缺canonical参数、未知/缺config、untied配置、head/embed不一致均拒绝。没有放松为一般missing-key白名单，也没有加载模型或修改训练。

新增两gate各3项CPU回归：合法tied duplicate通过、数值不一致拒绝、untied配置不能白名单化head，共6项；此前8项完整性/shape/NaN/reshard回归保留。3文件AST语法通过，扩充后61项目标CPU测试及实际第五步全部finite检查由root统一执行，本文此刻不提前宣称已通过。训练已继续第6步，验收脚本修正不停止trainer。
