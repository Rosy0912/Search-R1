# 实际执行后源码捕获

`r02_local_persistence_audit.executed.py` 是本次已执行的 Python stdin 原文，按工具调用字面源码在执行后持久保存。本次没有先保存、提交或发布该临时源码；不得把这个事后捕获当作 publication-before-run 已满足。没有再次执行该脚本。

原入口为当前 workspace 中 `python3 -B - <<'PY'`，stdin 即该文件原文，末尾 `PY` 关闭 heredoc；没有命令行额外参数。外层 tools.exec_command 使用默认 shell（当前 zsh），未显式指定 shell、workdir、login 或 sandbox_permissions。V8 functions.exec 只 await 此 exec_command 并输出其结果。

真实工具结果：exit_code=0；chunk_id=5cba57；wall_time_seconds=0.116875417。真实报告 SHA02746b0e6ebdb9fb3c8bac0cf82df78cd59a831fc1c6de6c76d15f5f8bf307c8，notes SHA150ef55d59cd3d984384ccd9e7d695e5786e6ffabc516127826be947400d3d01，derived manifest SHAda1aa2d9e3031008205155d471015e791844574012b0b3143b500701960e491d。

额外前置本地只读查看命令仅查看 scoped 路径和小 JSON、读取 frozen trainer 源码，未 Node/Git/模型/CUDA。一个前置小 receipt 打印程序曾对 files=39 的整数错误调用 len，真实 exit_code1（chunk1d4fa9）；这属于本地审读器接口错误，不是备份或训练故障。随后直接读取正确字段并由上述主核验 rc0 收尾。该失败不能改写为成功。

源码包含本工作区私有资产路径和身份哈希，没有凭据、问题正文、token 正文或权重字节，不含新增人工许可。脚本仅只读输入并写本地私有报告；没有远端命令、请求、信号、Torch 导入、模型载入或 Git。它目前是历史执行源码，不应未经 root 审阅发布后再次执行。
