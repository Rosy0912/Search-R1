# 在线 OPSD 当前控制器终态捕获

这是一个独立、一次性的 CPU 只读入口。它绑定当前在线 run
`1d1d67b60f0e46a2afc0536a36ab69b1`、不可变配置 `d0c30d…1b17`、
原 7cf helper、2cb wait recovery 和 51eb CPU merge recovery。它不改原源码、
配置、tmux 选项或路由，不启动模型、HTTP 检索、训练、GPU 工作或后台监控。
只有私有 JSON 写到 stdout；Root 负责源码发布、节点部署以及 stdout 证据保存。

`live` 在控制器还在运行时验证真实 UID/starttick/argv、前台 Bash 父子关系、
PGID、tmux server/socket/session/pane 身份和 `remain-on-exit=on`。它只解析
一条 `env … python … > 固定日志 2>&1` 命令，拒绝后续命令、管道、替换、
重定向变化或环境/argv 变化；不执行该字符串。真实启动 shell 与
`pane_start_command` 的 SHA 均须为 `5e424025…5a75`。
live stdout 的字节 SHA 及完整原文须由 Root 保存，用于之后的 terminal 绑定。
live 不声称已退出或 exit0。

`terminal` 接受此前 live JSON 的显式路径/SHA，或 `--binding -` 加 stdin
原字节 SHA。它不要求已死的旧 Python/Bash 仍可读取。必须是同一真实
tmux server UID/starttick、socket device/inode/ctime（不使用可变 mtime）、
session/pane 和启动命令，且保留的该 pane 实际 `pane_dead=1`、
`pane_dead_status=0`。旧 Python/Bash PID 仍存在或复用、server/socket/pane
消失/复用、空退出状态、COMPLETE 状态与身份不符等均为 UNKNOWN。

终态还要求原生 status 中同 owner/run/config 的 COMPLETE200、非零实际更新
计数和最新 checkpoint 引用。使用原已发布 V2 ownership scanner 的 freshly
SHA-checked 源字节直接 compile/exec，扫描所有同 UID 进程、run 环境、
真实 worker argv、父链和 opaque 进程。历史身份取自原 producer 实际写入的
`config.paths.output_root/child-history.jsonl`（runs），包括 sampling、training
和 CPU merge，及已绑定的旧 owner 退出票据。只豁免精确当前外部 CPU
capturer；未明来源的进程或后代仍阻断。不能使用要求 owner live 的旧 release
函数，也不能把缺 PID 当作 rc=0。

accepted terminal 只证明**这个精确控制器的真实 rc=0 与当次进程释放**。
COMPLETE200 字段不是 outer0–199 全部样本、K/Adam 累计、最终权重、备份或
效果的审计。历史失败 attempt 的退出码不由最终 wrapper0 推断；每个已提交
采样/训练 phase 的真实子进程退出码仍要核其 durable wait receipts。
新 native512 三权重效果评估属于后续独立入口，本目录尚未实现或启动它。

所有模式先核配置/launch/已发布完成与效果计划、自身 manifest/publication
及原三层源码的实际文件 SHA/bytes 和 remote-readback equal 证据。
这里不重 hash 6GB 模型、读取权重、推理、优化或恢复 RNG。源码/来源证据
发生变化时拒绝。UNKNOWN 的进程退出码为2；它表示证据未足够确认，不等于
正在训练的程序发生故障，更不会触发停止、重启或重新评测。

## 使用

先由 Root 发布本目录 README、两 Python 文件和 source-manifest，并生成
本目录 publication：accepted=true、branch=search-e1、source_commit 为40字
且等于 remote_readback_sha、run/config 相同、manifest SHA 相同、逐文件
relative repo path 的 SHA/bytes/equal 均通过。只提供历史 commit 名字不够。

以下是接口模板，必须填入实际已核 SHA，不是已运行的记录：

```text
env -u E1_RUN_ID CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  <原环境python> -B <发布目录>/capture_online_terminal.py live \
  --config <B>/receipts/config.json --config-sha256 <d0c30d完整SHA> \
  --launch-proof <实际启动票据> --launch-proof-sha256 <启动票据SHA> \
  --plan <已发布计划的原字节副本> --plan-sha256 <6b2af6完整SHA> \
  --source-manifest <发布目录>/source-manifest.json --source-manifest-sha256 <实际SHA> \
  --publication <本目录发布票据> --publication-sha256 <实际SHA> \
  --session e1-opsd-online-cpu-recovery2-200-20261008 --pane %70
```

终态使用相同全部选项，模式改 `terminal`，增加
`--binding <live原始JSON> --binding-sha256 <live原字节SHA>`。
capturer 不能继承训练 run 的 E1_RUN_ID。输入和输出均属于私有证据；本目录
Git inventory 不含 live JSON、启动 argv/env 正文、原始问题/轨迹/token 或权重。

## CPU 检查

```text
python3 -B -m unittest discover \
  -s experiments/search_e1/control/opsd_online_results_v1 -p 'test_*.py' -v
```

21 项 CPU fixtures 覆盖正确前台语法、额外命令/替换拒绝、进程与 socket/pane
复用、缺 tmux、空/非零实际 rc、死旧 owner 后的有效终态、COMPLETE 身份、
opaque/owned release 阻断、source/hash/readback、fresh scanner 无 pyc，及
原 producer 的 runs/child-history 路径负例。fixtures 不调用节点、真实 tmux
或 GPU。另已纯 CPU 解析 Root 提供的私有2014 live observation 与启动票据，
实际 shell SHA 与 pane 字节一致；这不等于新 CLI 已在节点完成实测。
