# Corrected eval-only/scoring-view 候选独立审查

日期：2026-10-04（Asia/Shanghai）。结论：固定候选的可选评分视图记录器、CPU replay 和 eval-only 守卫通过本次独立静态/CPU审查；**仍 NOT DEPLOYABLE，不是实际500、HF500或修复后效果验收**。没有修改远端3007/source8e，没有使用GPU，没有提交训练或评估作业。

## 固定审查快照

外部候选清单 work/reward-protocol-fix/corrected-eval-candidate-files.json 的 SHA256 为 c87372032b8e42a76925e1636a1c2214e94bafcbb097bf5c80a1c7343d6feb30。独立逐文件核验全部六个清单条目与实际字节一致。其中：
- corrected_evaluation.py：8d7b7fcfb5ee984878a712d112e69a813c7af1a2907f5703656ea4116a36643e
- baselines/main.py：8bd9233f63fdb3ce9be36c65fbc9db4132e7d51a53674581dadafcea287b8045
- ray_trainer.py：247583d565401ab6798e655829ca4ac5bcd7cfadf328fbb4259c1e0c743c4eaa
- test_corrected_evaluation.py：74b7cd2cc5cfc6d18c59b3073f9fce15c59d54f3a95e1a648ed0a8d4b535268f

审查基线 HEAD 为875b861472c6e020aab8d3b23dc2a35a04641032。上述新增/修改代码尚未以这个HEAD固定，必须由根任务完成新提交及原生检查后使用实际新评估commit；不能把875或5f标签当作已经涵盖记录器代码。

## 默认流程与评分一致性

默认 corrected_evaluation 未启用时 configure 立即返回，不切换数据集、不创建记录目录、不安装更新守卫。已有collect_qa仍调用同一个 batch_answer_metrics，reward/aggregate计算与generation循环未改变。新入口只在 prepare_common_holdout 已绑定原train/阶段64/final split之后切到原final7，避免重新构造不同split identity。

启用入口要求val_only、val_before_train、OPSD、do_search、同节点标签、val_batch_size64且无val_data_num。原始parquet文件及顺序、文件内row ordinal、prompt/source/全部target aliases在生成前由外部SHA绑定。load_inputs还读取真实parquet逐行核对，不能只重新哈希一个输出自报的identity。正式分母必须51713且七个source；start_capture验证过滤没有减题、SequentialSampler、drop_last=False。writer与replay都检查每批min(64,remaining)，即808×64、最后1。

writer在原生union/撤销DP补齐之后、原RewardManager计分之后读取DataProto，detach/CPU/clone保存原整数token和attention/info mask，并以真实DataProto重跑原scorer。逐题EM/F1/格式/生成token/search_calls、model-only spans、原reward和最终聚合严格一致才写完成manifest。不拼接跨observation片段，不以重新编码文本代替原token。生成前question/targets身份与原始输入逐条匹配，不从输出自证身份。

该对象明确为 **native-scoring-view-not-full-turn-trace**：现有生成器会截累计response，本候选不记录截断前全部turn，不得声称保存了完整工具轨迹。它保存的是原生评分实际使用的视图，没有改变截断、采样或评分预算。

## 来源和零更新门禁

阶段清单必须由外部OPSD_CORRECTED_EVAL_STAGE_SHA256绑定；main显式将变量传入Ray main_task runtime_env。十三个必要代码文件均要求byte SHA；其中十个CPU安全模块还核对实际import的__file__/SHA，generation和dataset包含在内。actor/distillation/vLLM worker源码这里只byte-pin，实际GPU worker进口路径/摘要仍由后续独立runtime supplement核验。

configure与start_capture均将实际resolved generation/retrieval预算逐字段对照外部stage；缺字段、预算改变均拒绝。预算包括原生vLLM do_sample=False下temperature0/n1/best_of1/top_p1/top_k-1/min_p0硬编码的来源。CPU replay也重新验证代码、外部stage、模型文件及原始输入链。

ZeroUpdateGuard替换实际worker-group update_actor/update_critic分发方法，记录原分发身份；尝试调用立即计数并拒绝，完成前检查守卫未被替换。replay验证guard源码SHA、真实actor dispatch项目、original_dispatches和零尝试计数。HF前后文件SHA是一项独立检查，不能替代这一调用守卫。

## 本次发现并修正的候选问题

实施agent在审查反馈后修正了：CLI先于新verify函数定义导致replay NameError；评分replay仅相信source标签；必要源码集合降为六个遗漏生成/数据/Ray hook；实际预算未与外部stage匹配；正式batch64/尾1缺少replay检查；Ray stage pin未显式传入；测试使用空格敏感字符串及旧缺题夹具不再符合批守卫。固定快照已核对这些修复。早期版本不在本次通过范围。

## 独立CPU执行证据

使用已存在的 /Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python，SEARCH_R1_REQUIRE_REAL_DATAPROTO=1、CUDA_VISIBLE_DEVICES为空、OMP_NUM_THREADS=1。独立执行当前固定字节的三个测试文件：
- tests/reproduction/test_corrected_evaluation.py
- tests/reproduction/test_reward_and_validation.py
- tests/reproduction/test_terminal_answer_protocol.py

实际结果：**66 passed, 2 warnings in 4.59s，exit0**。真实verl.protocol.DataProto；没有AST DataProto fallback。已有urllib3/LibreSSL与Ray API弃用警告，不影响CPU结果。实施agent单独原始日志66passed/4.07s也与候选清单SHA匹配；本次4.59s是独立重新执行结果，不混用旧233+5。

另独立执行9项纯证据负例（JSON重复key、NaN、overflow；六种mask/shape/dtype/token缺失问题）及合法数组/CLI定义顺序检查通过。使用真实DataProto/同固定writer另做五个独立replay负例：相对路径逃逸、symlink逃逸、重复批次、缺原始npz文件、旧8e来源冒充corrected，全部拒绝。篡改测试在适用时连manifest一起重新SHA，仍被相应结构/来源检查拒绝。

CPU raw-view单元测试明确只mock未来严格training/HF prerequisite边界，使用合成tokenizer及合成model内容文件；它们检验评分视图与门禁，不证明actual3007已完成500、HF435tensor/3shard有效或任何GPU性能。生产该边界保持真实缺失即blocked；另有直接调用未mock checker的缺前置工具/stage漂移负例。

## 仍未完成、不能跨越的边界

1. strict_training_hf_verifier尚未实现/独立审查，未来actual500/终态/periodic64/final51713/HF500完成receipt仍不存在。其空缺使生产load_stage/replay失败；一个打印匹配JSON的脚本不能替代真实完整证据检查。
2. 模型role/path必须由该独立严格桥核验：base绑定真实初始模型，hf500绑定actual3007的同一个完整HF500及全部文件摘要。当前verify_model内容SHA与role枚举不足单独认定角色，候选明确将此留在blocked桥，不能用自报字段授权部署。
3. 仍需Linux原生环境检查、实际原始七集有序manifest/完整dataset过滤核验、同节点GPU检索/租约/实际worker imports、实际0更新与清理、base/HF500各51713的新评估、全量CPU replay和独立paired-effect receipt。不得把合成65题CPU检查当作真实51713效果。
4. 原3007训练source8e、旧阶段/最终评分结果、旧HF500预览都保留。新评估source必须另固定，训练receipt与corrected receipt分角色绑定，旧分数不能伪装成新协议分数。

允许根任务在完成所需原生CPU检查后提交此**默认关闭且生产仍blocked的候选**，不据此启动GPU评估、填未来receipt或宣称效果完成。后续严格桥和runtime supplement需要新的独立审查。

## 原生结果与提交前空白归一化补审

已读取根任务保留的 Linux 原生检查结果与 NATIVE_REVIEW：隔离875 checkout覆盖原c873四文件并逐SHA核验，真实DataProto强制、CUDA不可见、OMP/MKL1，287 passed/22 warnings/28.63s/exit0；记录的原始测试logSHA为0d07960eafb51822d6b65d51f475eb75fe3e75d3313330136c9ce21f452bd77d。原训练父repo仍8e且tracked-clean。它补足本报告先前待完成的Linux CPU检查，不补足actual500/HF或GPU门禁。

独立验证归一化后的module SHA为dced486435251b6e47c039c6f1a7edadbdab4f55e142b8d1f96e967a74b829d3。将当前文件字节仅追加一个LF，恰好恢复原c873 module SHA8d7b7fcfb5ee984878a712d112e69a813c7af1a2907f5703656ea4116a36643e；包括全部line/column attributes的Python AST完全相同，其余三个源码/测试SHA保持原清单值。因此只去掉一个多余EOF空行，没有功能改变，不需仅为该空白重复287项测试。原清单及测试证据继续作为原始字节历史保留。

独立同意根任务curated提交默认关闭helper、测试和本次审查记录。此结论仅批准提交范围，**不批准部署GPU评估，不允许填actual500/HF500或paired效果receipt**；严格桥、实际角色与GPU worker/runtime证据继续blocked。
