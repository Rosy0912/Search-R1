# Tree-GRPO 跨 root 批处理修复

修复 `audit-20261002/tree-grpo.md:F3`：每个扩展轮次先从各 root 当前节点采父节点，再把所有 root 的 continuation 合批完成；新快照按所属 root 分别去重并登记，下一轮继续可采。完成后仍逐 root 选 K 叶。原 source/tree UID、复制的 prefix/observation、剩余 turn budget、叶路径重复训练和全部 padding 生成成本记录均保留。

修改：`search_r1/baselines/rollout.py:run`；专属回归 `tests/baselines/test_tree_batching.py`。未改 actor/controller/advantages、原 generation.py，也未启动远程任务。

验收：`.venv-cpu/bin/python -m unittest tests.baselines.test_tree_batching tests.baselines.test_algorithms.Rollouts -v`：9 tests OK。两 root/每 root 两 child/8 GPU 的 post-search 续写只产生一次 expansion worker call；总 worker call 从4降3，discarded padding从24降16。真实检索2次不重放、6叶来源/树/轨迹身份正确、steps/mask对齐。三root两round验证14 padding、15 sampled trajectories、每root节点快照仍2个、turn预算不重置。

局限：合批改变 RNG 消耗的 root/round顺序，因此相同 seed 未承诺与旧实现产生逐项相同抽样叶；每 root 有放回父节点采样与无放回选叶分布仍相同。实际GPU vLLM采样也可能依batch组成改变随机实现。本次证明语义/账本回归，不证明正式3B吞吐或论文节省率。
