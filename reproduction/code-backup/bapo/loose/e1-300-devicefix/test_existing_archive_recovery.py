"""Real tiny archive refusal tests; not actual3156 recovery."""
import json,subprocess,tempfile,tarfile,unittest
from pathlib import Path
SCRIPT=Path(__file__).with_name('recover3156-existing-archive-worker.py')
class ArchiveRecovery(unittest.TestCase):
 def setup(self,base):
  root=(base/'owned').resolve();(root/'logs').mkdir(parents=True);(root/'runs/job').mkdir(parents=True);(root/'repo').mkdir()
  subprocess.run(['git','init','-q',str(root/'repo')],check=True)
  (root/'repo/file').write_text('real local source')
  subprocess.run(['git','-C',str(root/'repo'),'add','file'],check=True)
  subprocess.run(['git','-C',str(root/'repo'),'-c','user.name=fixture','-c','user.email=fixture@local','commit','-qm','fixture'],check=True)
  source=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()
  (root/'logs/tools-manifest.json').write_text(json.dumps({'files':{str(i):'fixture'for i in range(21)}}))
  p=root/'runs/job/raw.txt';p.write_bytes(b'real preserved bytes');archive=root/'logs/hardware-3156-raw-capture-fixture.tar.gz'
  with tarfile.open(archive,'x:gz')as tf:tf.add(p,arcname='runs/job/raw.txt',recursive=False)
  text=SCRIPT.read_text().replace("Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-countfix-20261004')",'Path('+repr(str(root))+')').replace("SOURCE='f0e37bf68d61ec926ac225f5e1ee1f606cb09bf9'",'SOURCE='+repr(source))
  # Keep tests independent of a real3156 process; archive file logic executes unchanged.
  text=text.replace('active=writers();assert not active,repr(active)','active=[];assert not active').replace("assert not writers(),'Original writer appeared; no recovery accepted'",'assert True')
  script=base/'worker.py';script.write_text(text);return root,p,archive,script
 def run_script(self,root,script):
  subprocess.run(['python3',str(script)],check=True,capture_output=True)
  return json.loads((root/'logs/hardware-3156-existing-archive-recovery-result.json').read_text())
 def test_real_tar_current_sha_verified_without_new_archive(self):
  with tempfile.TemporaryDirectory()as d:
   root,p,a,s=self.setup(Path(d));before=a.read_bytes();v=self.run_script(root,s)
   self.assertTrue(v['no_new_archive']);self.assertEqual(v['files'][0]['bytes'],len(p.read_bytes()));self.assertEqual(a.read_bytes(),before)
 def test_changed_current_file_refused_and_oldarchive_preserved(self):
  with tempfile.TemporaryDirectory()as d:
   root,p,a,s=self.setup(Path(d));before=a.read_bytes();p.write_text('changed current');v=self.run_script(root,s)
   self.assertEqual(v['status'],'RECOVERY_REFUSED_NO_ARCHIVE_OR_PROCESS_CHANGED');self.assertEqual(a.read_bytes(),before)
 def test_unsafe_archive_member_refused(self):
  with tempfile.TemporaryDirectory()as d:
   root,p,a,s=self.setup(Path(d));a.unlink()
   with tarfile.open(a,'x:gz')as tf:tf.add(p,arcname='../outside',recursive=False)
   self.assertEqual(self.run_script(root,s)['status'],'RECOVERY_REFUSED_NO_ARCHIVE_OR_PROCESS_CHANGED')
 def test_ambiguous_or_corrupt_existing_archives_refused(self):
  for kind in ('ambiguous','corrupt'):
   with tempfile.TemporaryDirectory()as d:
    root,p,a,s=self.setup(Path(d))
    if kind=='ambiguous':a.with_name('hardware-3156-raw-capture-second.tar.gz').write_bytes(a.read_bytes())
    else:a.write_bytes(b'incomplete gzip')
    self.assertEqual(self.run_script(root,s)['status'],'RECOVERY_REFUSED_NO_ARCHIVE_OR_PROCESS_CHANGED')
if __name__=='__main__':unittest.main()
