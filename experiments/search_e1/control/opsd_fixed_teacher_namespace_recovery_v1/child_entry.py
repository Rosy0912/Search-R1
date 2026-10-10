import sys,runpy
from bootstrap import apply,H
apply()
assert sys.argv[1]==str(H/'trainer.py')
sys.argv=sys.argv[1:]
runpy.run_path(str(H/'trainer.py'),run_name='__main__')
