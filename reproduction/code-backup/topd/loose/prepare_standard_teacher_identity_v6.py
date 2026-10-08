"""Isolated source-only identity repin; no runtime/model/GPU operation."""
from pathlib import Path
import shutil,hashlib
W=Path('work');old='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0';new='12da9b40250360f5a9448d546c93949186eed2e35bf80be9250a1b5e1db338be';oldcommit='0d8a9214dc6e19f10e21549b56f6c01b45d67100';commit='e623724532d06e975b61f3b35ec7925dc790c939'
names=['qwen25-32b-trained-eval64','topd-32b-native-score-gate','topd-32b-shared-owner','topd-32b-pipeline-ipc','topd-shared32b-supervisor']
for name in names:
 src=W/(name+'-r1-v5');dst=W/(name+'-r1-v6');assert not dst.exists();shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
 for p in dst.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json','.md','.sbatch'):
   p.write_text(p.read_text().replace('-r1-v5','-r1-v6').replace(old,new).replace(oldcommit,commit).replace('0d8a921','e623724').replace('manifest9040','manifest12da'))
e=W/'qwen25-32b-trained-eval64-r1-v6';p=e/'training_identity.py';s=p.read_text().replace("KIND='SEARCH_R1_GRPO100_32B'", "TOOLS_MANIFEST='abf316a0018dd36c09a46162e8219033b0c66f53b937498c6cd826cac71a0ef8'\nKIND='SEARCH_R1_GRPO100_32B'")
s=s.replace(" args=config.get('training_command',[])"," need(config.get('tools_manifest',{}).get('sha256')==TOOLS_MANIFEST and review.get('tools_manifest_sha256')==TOOLS_MANIFEST,'Actual latest standard100 reviewed tools identity required')\n args=config.get('training_command',[])")
s=s.replace("'trainer.save_freq':'100'}", "'trainer.save_freq':'100','trainer.weight_continuation_step':'0'}")
s=s.replace(" need(all(options.get(k)==v for k,v in expected.items()),'Actual standard100 recipe geometry differs')", " need(all(options.get(k)==v for k,v in expected.items()),'Actual standard100 recipe geometry differs')\n need(options.get('actor_rollout_ref.model.actor_checkpoint_path','null')=='null','Fresh official base cannot load an actor checkpoint')")
p.write_text(s)
p=e/'test_training_identity.py';s=p.read_text().replace("source_manifest_sha256=m.SOURCE_MANIFEST,actual_update_evidence", "source_manifest_sha256=m.SOURCE_MANIFEST,tools_manifest_sha256=m.TOOLS_MANIFEST,actual_update_evidence")
s=s.replace("'trainer.save_freq':'100'}", "'trainer.save_freq':'100','trainer.weight_continuation_step':'0'}").replace("source_manifest={'sha256':m.SOURCE_MANIFEST},training_command", "source_manifest={'sha256':m.SOURCE_MANIFEST},tools_manifest={'sha256':m.TOOLS_MANIFEST},training_command")
s+='''\n@pytest.mark.parametrize('case',['old_branch','old_tools','mismatched_review_tools','continuation','actor_checkpoint'])
def test_latest_freshbase_identity_rejects_old_or_resume(case):
 a=records()
 if case=='old_branch':a[0]['source_reference']='0d8a9214dc6e19f10e21549b56f6c01b45d67100'
 if case=='old_tools':a[0]['tools_manifest_sha256']=a[3]['tools_manifest']['sha256']='ad422ea17014180a88500446faf73175908b3c2fad9a1c8bd50e071ca51fa68d'
 if case=='mismatched_review_tools':a[0]['tools_manifest_sha256']='0'*64
 if case=='continuation':a[3]['training_command']=[x.replace('weight_continuation_step=0','weight_continuation_step=2') for x in a[3]['training_command']]
 if case=='actor_checkpoint':a[3]['training_command']+=['+actor_rollout_ref.model.actor_checkpoint_path=/own/profile2']
 with pytest.raises(RuntimeError):m.check_standard100(*a)
''';p.write_text(s)
p=e/'trained_assets.py';s=p.read_text().replace('96e60ceb5f877f1e4c6c467a5a1a84ba4a536ad0ad1bfcccadb3015cb03e7c6c',hashlib.sha256((e/'training_identity.py').read_bytes()).hexdigest());p.write_text(s)
p=W/'refresh_standard_teacher_identity_v6_pins.py';p.write_text((W/'refresh_standard_teacher_identity_v5_pins.py').read_text().replace('-r1-v5','-r1-v6'))
p=W/'test_standard_teacher_identity_v6_chain.py';s=(W/'test_standard_teacher_identity_v5_chain.py').read_text().replace('-r1-v5','-r1-v6').replace(old,new).replace('source-v2/','source-v4/').replace("'search-r1-32b-grpo100-source-v2'", "'search-r1-32b-grpo100-source-v4'").replace('r1-32b-hf-semantic-review-v4','r1-32b-hf-semantic-review-v5');p.write_text(s)
