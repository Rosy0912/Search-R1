import json
import shlex
import subprocess
import time

JOB_NAME = "bapo-exclusive-hold-20261012"
DEADLINE = 1791774000

def run(command):
    p = subprocess.run(command, shell=True, text=True, capture_output=True)
    return {"command": command, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

remote = (
    "existing=$(squeue -h -n " + shlex.quote(JOB_NAME) + " -o %i); "
    "if [ -n \"$existing\" ]; then scontrol show job \"$existing\"; exit 0; fi; "
    "id=$(sbatch --parsable "
    "--job-name=" + JOB_NAME + " "
    "--partition=gpu --nodelist=h100-11 --nodes=1 --ntasks=1 "
    "--cpus-per-task=96 --gres=gpu:h100:8 --mem=768G --exclusive "
    "--time=01:00:00 "
    "--output=/dataset/shaoshitong/search-r1-rl/bapo/logs/slurm-exclusive-hold-%j.out "
    "--error=/dataset/shaoshitong/search-r1-rl/bapo/logs/slurm-exclusive-hold-%j.err "
    "--wrap=" + shlex.quote(
        "python3 -c 'import time; time.sleep(max(0, " + str(DEADLINE) + "-time.time()))'"
    ) + "); "
    "case \"$id\" in (''|*[!0-9]*) echo \"invalid sbatch id: $id\" >&2; exit 2;; esac; "
    "scontrol show job \"$id\""
)

result = run("ssh -o BatchMode=yes -o ConnectTimeout=12 h100-5 " + shlex.quote(remote))
print(json.dumps({"time": time.time(), "job_name": JOB_NAME, "deadline": DEADLINE, "result": result}, ensure_ascii=False))
raise SystemExit(result["returncode"])
