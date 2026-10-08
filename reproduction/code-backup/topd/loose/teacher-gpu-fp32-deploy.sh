set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
sha256sum topd-teacher-gpu-fp32-tools.tar.gz
tar -xzf topd-teacher-gpu-fp32-tools.tar.gz -C tools
sha256sum tools/topd-teacher-gpu-fp32-probe.py tools/topd-teacher-gpu-fp32-probe.sbatch
record=logs/teacher-gpu-fp32-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/topd-teacher-gpu-fp32-probe.sbatch > "$record"; cat "$record"; fi
