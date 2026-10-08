# H20 环境重建

适用范围：Linux x86_64、CPython3.10、CUDA驱动支持Torch2.4的cu121 wheel。此机器的H20兼容设置沿用已实测环境，不安装或替换系统Torch/驱动。`requirements-h20.txt` 固定23个直接需求，`constraints-h20.txt` 固定其155项活跃依赖闭包（从隔离venv的Requires-Dist递归取得，不是盲复制系统freeze）。这是版本约束，不是带所有wheel哈希的供应链lock。

把审查后的仓库放在 `/data/search-r1/repo`，并把 bootstrap.sh、check_environment.py、两份requirements/constraints复制到项目脚本目录，随后在远端执行：

```bash
bash /data/search-r1/bootstrap.sh
```

脚本不checkout已有仓库，检查其含固定上游commit，并记录当前HEAD和工作树差异。创建缺失venv；已有Python不是3.10时停止。它使用与pipeline相同的flock，已有任务占用时不修改环境。根盘紧张，uv、pip、HF、Ray/运行脚本所需数据均应留在项目 `/data` 分区。

安装顺序：Torch2.4/vLLM0.6.3/全部固定依赖→校验官方FlashAttention wheel并no-deps安装→no-deps安装已审查仓库→**最后**no-deps安装cuBLAS12.4.5.8。Torch原声明固定cuBLAS12.1.3.1，与最终H20修复不一致，不能把12.4版本放在同一resolver事务。`check_environment.py` 只允许这一项声明例外，拒绝其它缺包/冲突，记录JSON；uv pip check原始输出也保留。不得在override之后再运行未受控的依赖安装，否则可能把cuBLAS降回崩溃版本或升级Torch。

FlashAttention官方wheel：`flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`；SHA256：`a8719a86d0cd12a105739151f9e45ac8c2ecb840a83ed7a080a88d1645a98440`。来源为[Dao-AILab官方release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1)，对应Torch2.4、Python3.10、ABI FALSE。本次既有安装的完整wheel已成功验SHA，可直接复用。

优先复用目标 `assets/` wheel和 `wheelhouse/` 内依赖wheel；没有Flash wheel时先目标直接下载，15秒连接超时/60秒无进度即失败退出；失败partial独立保留，不删除。已有错误完整wheel改名 `.invalid.<时间>` 保留。默认PyPI源是已在目标验证过的Aliyun镜像，可通过 `SEARCH_R1_PYPI_INDEX` 显式选择官方PyPI；未改变用户全局pip/SSH设置。

## 下载不联网时的Mac中转

在Mac执行以下单目标示例。沿用现有 jump_h20-1 SSH路由；只有远端最终SHA核验、移动成功后才删除本次Mac临时副本。任何失败都会使set-e停止并保留副本。多目标时必须全部验证之后再清理。

```bash
set -e
mkdir -p "$HOME/Documents/remote_codex_control/.transfer-staging"
search_r1_staging=$(mktemp -d "$HOME/Documents/remote_codex_control/.transfer-staging/Search-R1.XXXXXX")
search_r1_wheel=flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
curl -fL --connect-timeout 15 --speed-time 60 --speed-limit 1024 \
  -o "$search_r1_staging/$search_r1_wheel" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$search_r1_wheel"
python3 - "$search_r1_staging/$search_r1_wheel" <<'PY'
import hashlib,sys
assert hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest() == 'a8719a86d0cd12a105739151f9e45ac8c2ecb840a83ed7a080a88d1645a98440'
PY
scp -o ConnectTimeout=15 "$search_r1_staging/$search_r1_wheel" \
  "jump_h20-1:/data/search-r1/assets/$search_r1_wheel.incoming"
ssh jump_h20-1 'set -e; cd /data/search-r1/assets; printf "%s  %s\n" a8719a86d0cd12a105739151f9e45ac8c2ecb840a83ed7a080a88d1645a98440 flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl.incoming | sha256sum -c -; mv flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl.incoming flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python3 - "$search_r1_staging/$search_r1_wheel" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);p.unlink();p.parent.rmdir()
PY
```

连接可能因VPN或JumpServer变慢；失败副本仍在输出的本次目录中，重新传输即可，不改服务器上的凭据。

完全离线的依赖安装需事先在 `wheelhouse/` 中备齐 Linux/Python3.10 的完整依赖wheel及cuBLAS12.4.5.8，并按下载记忆先目标下载、不通才Mac中转、所有目标核验后清理；Mac原生arm64 wheel不能代替Linuxwheel。`BOOTSTRAP_OFFLINE=1 bash bootstrap.sh` 会禁用索引/network依赖获取；缺wheel明确失败，不默默回退联网。Torch标准12.1依赖wheel和最后12.4override wheel都需备齐。仓库源码本身也必须先传入。

## 验收范围

`environment-acceptance.json`、`environment-dependency-check.txt`、`environment-freeze.txt`、`environment-source-commit.txt` 与差异记录可审计安装结果。`environment.ready` 仅代表固定版本、导入及允许的声明例外通过；**不代表GPU兼容或论文复现成功**。唯一训练pipeline还必须完成8卡NCCL/BF16、真实推理、检索、真实短训练及checkpoint/最终EM检查。训练脚本继续使用 `NCCL_NVLS_ENABLE=0`。

固定训练环境已在独立目录真实重建，并完成8卡NCCL/BF16及真实3B推理；结果见 `../acceptance.md`。首次在其它环境重建仍必须执行集成门禁。

## 隔离的 GPU 检索 overlay

另运行 `bash /data/search-r1/bootstrap_retriever.sh`，将 `faiss-gpu-cu12==1.8.0.2` 装到项目的 `retriever-gpu-lib`，只有retriever及其preflight使用该PYTHONPATH，不替换训练venv中的faiss-cpu、Torch或NumPy。该PyPI wheel基于FAISS1.8.0，来自第三方faiss-wheels构建，不是Meta官方二进制发行；固定cp310/Linux x86_64包SHA256为 `7ea5e6aa82897fd0c1af3b1decc9c8659a4398fed42de1d7746715b64bbf9bf9`。它沿用当前cuBLAS12.4.5.8和NumPy1.26.4；依赖解析禁用，避免触动训练环境。PyPI该旧版本曾因NumPy2兼容性被yank，此项目明确使用NumPy1.26.4，并已实际导入和GPU索引验收。

下载优先目标远端，超时后按下载记忆从Mac中转，同样先验证SHA再安装。脚本使用项目锁，已有训练时拒绝修改overlay。上游 `--faiss_gpu` 参数启用8卡分片FP16索引；real-index preflight比较14个跨源真实问题的CPU/GPU top3，并计时512条索引查询。该检查仅有限样本数值核对，不证明全量排序一致。
