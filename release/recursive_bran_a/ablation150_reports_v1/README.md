# 150轮消融小型汇总

按最新人工范围只汇总完整dev512的overall/NQ/Hotpot EM，与A同轮次总体EM差值；**不追加逐题捕获、bootstrap或A重评测**。工具只读已有文件，纯Python标准库，无模型、Torch、GPU、HTTP或训练锁。原始轨迹与checkpoint不复制到Mac。

```sh
python -B release/recursive_bran_a/ablation150_reports_v1/report.py \
  --study /远端/本任务/receipts/report-study.json \
  --output /远端/本任务/reports/summary
```

生成三份小文件：`aggregate.json`、`curve.csv`、`findings-zh.md`。未发生的5/50/100/150节点保留`MISSING`，不以部分预测、TRAIN reward或历史整数补结果。`analysis_complete`仅指指定汇总节点的EM齐全，不代表150训练终态、资源释放或整个goal完成。

`report-study.json`示例（所有路径换为本任务实际远端路径）：

```json
{
  "reference_condition": "A",
  "milestones": [5, 50, 100, 150],
  "conditions": [
    {
      "id": "A",
      "run_id": "对应历史实际run",
      "config_sha256": "对应历史实际配置SHA",
      "world_size": 8,
      "logs": ["/原A/保留的有效training.log"],
      "output_roots": ["/原A/保留的有效output"],
      "aggregate_files": {
        "150": "/原A/reports/h100-outer150-20261014/aggregate.json"
      }
    },
    {
      "id": "C1",
      "run_id": "本组实际run",
      "config_sha256": "本组实际配置SHA",
      "world_size": 8,
      "logs": ["/本组/logs/training.log"],
      "output_roots": ["/本组/output"],
      "memory_snapshot": "/本组/control/memory-live.json"
    }
  ]
}
```

`reference_condition`可省略；条件之间不共享或猜测run/config。`logs`和`output_roots`必须明确指向保留的有效训练链，不能把回退废弃分支混入。相同outer出现冲突日志或原生收据会拒绝汇总，不采取“最后一条赢”。恢复时可列出多个有效前缀/后缀目录；相同outer的等内容证据自动去重。若A只保留公开aggregate，`aggregate_files`支持现有`dev512.{count,hits,nq,hotpot}`格式，不声称有逐题证据。

输入与计数：

- 完整原生`step:N - ...`中的`val/qa_count/*`和`val/em/*`，核512和NQ/Hotpot各256、整数hits与逐集守恒。
- `output/attempt-results/attempt-*.json`中的原生all-rank result，核实际20机会、真实写入与跳过计数；所有前缀收据覆盖时才给累计Adam/skips。缺失收据时仅单独列原生日志的累计Adam，不冒充逐rank完整实读。
- `rollout/raw_emitted_tokens`优先，兼容`rollout/generated_tokens`。计数依原生约定包含padding与丢弃尾部；覆盖缺失明确标记，不把已记录部分称全训练成本。
- `timing_s/*`为原生阶段标量；分量可能重叠，不相加当总耗时。GPU小时为记录的outer wall × 卡数，不是忙碌GPU小时或整个Slurm租期。
- `output/recursive-events/*/*/events.jsonl`只统计已闭合的稀疏记录，非记录节点未知；没有重读轨迹tensor、重建LCP或μ。`memory_snapshot`只保留原observer的采样峰值语义。

最小CPU检查：`python -B -m unittest discover -s release/recursive_bran_a/ablation150_reports_v1 -p 'test_*.py'`。
