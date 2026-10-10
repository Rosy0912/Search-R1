"""Original native trainer with the separately published operational outer guard repaired."""
import sys,runpy
from bootstrap import H
target=sys.argv[1];assert target==str(H/'trainer.py');sys.argv=[target]+sys.argv[2:];runpy.run_path(target,run_name='__main__')
