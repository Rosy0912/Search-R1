"""Map frozen teacher's owned root to HPC3; retain strict path checks."""
import ast,subprocess
from finalize_topd200_draft import ROOT,main as refresh
if 'topd' in subprocess.check_output(['squeue','-u','mou176','-h','-o','%j'],universal_newlines=True).lower():raise RuntimeError('Active TOP-D job')
p=ROOT/'tools/topd-vllm-teacher-v3/topd_vllm_teacher/daemon.py'
s=p.read_text().replace('/dataset/shaoshitong/search-r1-rl/topd',str(ROOT));ast.parse(s);p.write_text(s)
refresh();print('TEACHER_HPC3_OWNED_ROOT_MAPPING_PASS')
