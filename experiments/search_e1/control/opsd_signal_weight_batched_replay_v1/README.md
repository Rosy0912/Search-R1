# 已有批量512响应的概率回放

不生成新轨迹，不搜索，不训练，不调整损失。全部100轮及九个同协议端点评测实际完成后，adapter.py读取v3的固定参考bank、实际八分片和canonical trace，在HKUST4 analysis下形成新带类型的回放输入。新vLLM结果不伪装成旧HF native512；旧HF239 bank不复用。原bank条目不变，只补study/config命名空间并保留原bank SHA。

replay_math.py、replay_worker.py和summary.py与已发布235e376分析代码逐字节相同；replay.py只更换plan类型，common.py只接受新的typed batched endpoint并重新确认严格评分及全512覆盖。CPU准备也要求四组训练真实终态；GPU入口继续用原资源锁、四卡及固定raw权重，无新增优化器或采样。源/端点输入兼容代码的CPU检查不能当作实际GPU完成。

每个端点保留512 EM，包括baseline无成功参考题；无B时teacher/g/KL为NA，普通student entropy不依赖B。反向KL及clip10全词表标量、ordinary source漂移、final answer概率差和think/search/answer区域贡献复用原公式。回放是HF BF16 base/FP32 raw LoRA、原响应重新编码的未处理概率，与vLLM合并BF16生成时的processed score和历史在线loss严格区分。不把不同响应逐token配对，不把无效标签字面答案当EM。所有完整token数据留远端。

完成后使用summary.py汇总各端点EM与g/h/KL相关、排序AUC、覆盖及同题新增/丢失；常量/单一EM类别为NA，区域loss不声称参数梯度或因果。正式EM主对照仍由已发布batched_summary_v1按原两项比较及Holm/配对bootstrap处理。

运行顺序：CPU adapter.py --request PATH --request-sha256 SHA --output REMOTE_NEW_DIR；读取其plan.json后，独立E1_RUN_ID和CVD空CPU parent执行replay.py --plan PATH --plan-sha256 SHA --output REMOTE_NEW_REPLAY_DIR。准备request须绑定全部四config、原四组真实终态、v3 complete、此helper/root/manifest/publication和新analysis_run_id。未来真实执行失败保留partial，在独立修复副本仅恢复缺项，不自动重跑生成或训练。
