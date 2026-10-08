# 七组十轮消融：C0 首轮恢复与直连

最新人工答复已明确允许启动，HKUST 本地输出和后台 Mac 持久备份；备份完成不作为训练启动条件。2026-10-09 再次确认本机直连 HKUST。实际 `ssh -B en0 -oProxyCommand=none -oProxyJump=none` 登录 HKUST1/HKUST4 成功；未修改全局 SSH/VPN 或凭据。

首个独立顺序 driver 6566ff8f6da34caf8f2484d24f431a64 的实际 pane%71 返回2；C0 控制器实际返回1并已完整释放自有 GPU 子进程。C0 首轮完整2560轨迹、327原规则配对、教师诊断已完成；训练在首个Adam前退出，没有已提交checkpoint。当前不能称已有更新或效果验收。原源码871dc46、配置、采样/HTTP/轨迹、失败日志和退出证据均保留。

PEFT0.13.2 新增 `outer_old` adapter继承BF16，而共同初始及轮次checkpoint适配器为FP32。完整摘要含dtype，身份检查正确拒绝。实际同环境小模型CPU、零forward/零CUDA复现确认；非BF16可表示的更新权重表明只在加载后转FP32不足。独立修复launcher在原setter加载前仅将outer_old参数提升FP32，随后加载原FP32文件，保留504张量精确摘要、冻结版本、当前适配器和优化器检查。

新外部控制副本不修改七组原科学配置。仅准确替换原八rank torchrun的trainer脚本路径；C0使用显式同run `--resume` 复用已验收采样、配对和诊断，尚无Adam可恢复。其他六组保持fresh。新study控制器先确认旧pane真实rc2、旧进程身份全退出、零checkpoint、原outer0身份和新鲜内存，再保存原CPU role票据，迁移死driver身份，复用原鲜活observer。原失败不算成功终态。

修复后的每组仍十个在线outer，随后立即原协议native512。新真实更新、八rank状态、完整十轮以及效果分别验收；未启动或后台初始化不能冒充完成。三个窄控制边界测试、两个overlay测试、十三备份固定身份测试各自通过，不重复整套历史fixture，也不新采样测试后归零。

独立outer-end备份队列只处理已完成的固定checkpoint/完成票据，空间/网络失败写BACKLOG，不能停止或阻塞训练。旧CP89本地完整校验已完成；旧CP374和full192传输因代理中断保留partial，后续按直连独立恢复，不把包装进程0误称传输成功。HKUST4 archive原自有4–7继续，其他进程不触碰。
