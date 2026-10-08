"""Real subprocess/flock submission counterexample; fake scheduler, no SSH/GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).parent / 'training-tools/submit-training.sh'

class Submission(unittest.TestCase):
    def test_two_phases_cannot_submit_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();head=root/'head';head.mkdir();binpath=root/'bin';binpath.mkdir()
            script=SCRIPT.read_text().replace('/dev/shm/bapo-rewardfix-b097-scheduler-20261004',str(head))
            target=head/'submit-training.sh';target.write_text(script)
            for name in ('smoke2','accept5','formal500'):(head/(name+'.sbatch')).write_text('fixture')
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in head.iterdir() if p.is_file()}
            (head/'training-submit-approved-hashes.json').write_text(json.dumps(hashes))
            stubs={
                'flock':'import fcntl,sys;fcntl.flock(int(sys.argv[-1]),fcntl.LOCK_EX | (fcntl.LOCK_NB if "-n" in sys.argv else 0))',
                'ssh':'import time;time.sleep(.2)',
                'squeue':"import os,pathlib;p=pathlib.Path(os.environ['TEST_ROOT'])/'submitted';print('bapo-rfix-smoke2-b097' if p.exists() else '')",
                'sbatch':"import os,pathlib;p=pathlib.Path(os.environ['TEST_ROOT'])/'submitted';p.write_text(p.read_text()+'submit\\n' if p.exists() else 'submit\\n');print('9001')",
                'scontrol':"print('synthetic scheduler')"}
            for name,body in stubs.items():
                p=binpath/name;p.write_text('#!'+sys.executable+'\n'+body+'\n');p.chmod(0o755)
            env=dict(os.environ,PATH=str(binpath)+':'+os.environ['PATH'],TEST_ROOT=str(root),SMOKE2_JOB_ID='1')
            a=subprocess.Popen(['bash',str(target),'smoke2'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            b=subprocess.Popen(['bash',str(target),'accept5'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            outa=a.communicate(timeout=10);outb=b.communicate(timeout=10)
            self.assertIn(sorted([a.returncode,b.returncode]),([0,1],[0,3]),(outa,outb))
            self.assertEqual((root/'submitted').read_text(),'submit\n')

if __name__=='__main__':unittest.main(verbosity=2)
