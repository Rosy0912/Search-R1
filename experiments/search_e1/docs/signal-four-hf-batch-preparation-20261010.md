# 四组训练继续与原精度批量评测准备

2026-10-10 23:40左右CST实测：G_POS25/25，G_NEG21/25，H_POS/H_NEG尚未开始，总TRAIN46/100。G_NEG同一run/controller继续第22轮，从CP208采样，trace约4.02MB；不是重启或额外评测。原supervisor2297236/starttick693190113、controller2300905/starttick693209228保持，补测不作为后续训练或下一组启动门槛。

G_NEG第20轮P=K295，末Adam198，更新前TRAIN1153/2560=45.0390625%；第21轮P=K303，末Adam208，更新前TRAIN1191/2560=46.5234375%，累计pair曝光6352。每条TRAIN包括全错题和未选轨迹，不是更新后或固定测试集结果。CP198/208均已HKUST4→HKUST1各9成员逐SHA备份、两端实际exit0、Mac payload0。原生checkpoint与备份不冒充每轮独立Adam全张量CPU实读。

当次observer鲜活1.18秒，当前四卡directfree最小27529MiB、oracle718MiB。外来GPU0–3任务没有停止或修改。正式15/25补测仍0/8，全部100轮完成后补齐；训练科学设置、四rank累积8/global32不变。

旧补测慢的实际原因是四卡HF逐题生成，每张卡一次1题；训练使用vLLM批量采样。读取训练进度本身不需要任何补测。本次准备独立HF批量入口，不以CPU合并BF16/vLLM替代原精度。

原精度入口源码902b97ff3b21299d64578f30fe5ca7f59a56c643已提交推送、逐文件远程readback并独立部署hf-batch-eval-902b97f。保持HF4.46.3 SDPA/BF16底座和未合并的实际FP32 LoRA，逻辑rank seed42、8个64题分片，4物理卡两波、每卡batch32。原per-query HTTP、E5请求形状、8192context/500token每轮/5turn/4search/top3/480obs及严格评分保持。只有模型forward批量化；原执行器的各题停止与预算独立。

Node原环境16项CPU fixtures及微型随机Qwen2真实HF CPU生成对照1项通过，无下载、无实验权重加载、CUDA未初始化。原GP15实际rawCP150身份读取真实CPUwait0，完整raw cursor/权重digest保留；这不代表正式GPU评测或真实提速完成。此前46850d8候选缺config SHA的身份读取由902b97f独立副本修正，旧副本保留、未运行GPU，不是训练失败。

完整100训练真实exit0与资源释放后，再测试新同协议共同zero和8个raw15/25。批量padding/GEMM形状可能改变舍入，不强迫复现历史逐题HF239；全部端点统一使用新批量协议。每个worker保存实际生成ID、完整检索、模型加载/生成/总耗时、严格EM和新增/丢失，以真实运行验收速度和结果。

同精度最终消费者224b795已发布/readback/独立CPU部署。仅适配协议及loaded504个FP32张量身份，replay_math.py/replay_worker.py与旧已发布代码逐字节相同；7项CPUfixtures通过。首次Nodefixture缺repository sibling result_analysis搜索路径，已使用原已发布源码PYTHONPATH复跑通过，不修改生产源码或正跑。

后续以新同协议zero成功参考bank分析EM与gap/entropy/KL及区域贡献，普通entropy覆盖无参考题，teacher指标缺参考明确NA。概率回放是posthoc重编码的未处理概率，不冒充实时训练loss或生成时processed score。原两项25轮主对照、Holm及paired bootstrap数学不变；训练s/5关联与测试EM分开。无新增训练或额外采样；大产物全部HKUST远端。
