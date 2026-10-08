import sys
if sys.flags.optimize:raise RuntimeError("optimized Python forbidden")
from diagnostic_contract import verify,PHASE
if len(sys.argv)!=2 or sys.argv[1]!=PHASE:raise ValueError('Only zero-Adam HF probability diagnostic phase allowed')
verify()
print('ACTUAL_DIAGNOSTIC_CODE_CPU_INPUT_GATE_OK_NOT_NATIVE_ACCEPTANCE')
