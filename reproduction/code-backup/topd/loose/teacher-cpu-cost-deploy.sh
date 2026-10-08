set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
sha256sum topd-teacher-cpu-cost-tools.tar.gz
 tar -xzf topd-teacher-cpu-cost-tools.tar.gz -C tools
sha256sum tools/topd-teacher-cpu-cost-probe.py tools/topd-teacher-cpu-cost-probe.sbatch
record=logs/teacher-cpu-cost-first-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/topd-teacher-cpu-cost-probe.sbatch > "$record"; cat "$record"; fi
