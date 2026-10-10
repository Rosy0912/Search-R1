"""Bound teacher prompt-logprob temporary tensors without changing scoring scope."""
import ast,subprocess
from finalize_topd200_draft import ROOT,main as refresh
if 'topd' in subprocess.check_output(['squeue','-u','mou176','-h','-o','%j'],universal_newlines=True).lower():raise RuntimeError('Active TOP-D job')
p=ROOT/'tools/topd-vllm-teacher-v3/topd_vllm_teacher/daemon.py';s=p.read_text()
old='max_num_batched_tokens=max(a.max_model_len,8192)'
new='max_num_batched_tokens=2048'
if old not in s:raise ValueError('Unexpected teacher prefill configuration')
s=s.replace(old,new);ast.parse(s);p.write_text(s);refresh()
print('TEACHER_CHUNKED_PREFILL_2048_SET_FULL_SEQUENCE_AND_SCORING_UNCHANGED_GPU_VALIDATION_PENDING')
