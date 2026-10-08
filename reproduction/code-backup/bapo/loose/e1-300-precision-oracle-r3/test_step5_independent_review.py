"""Actual tokenizer semantic refusal tests; these do not fake HF model acceptance."""
import importlib.util,tempfile,unittest,json,copy
from unittest.mock import patch
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
spec=importlib.util.spec_from_file_location('step5review',Path(__file__).with_name('step5-extract.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ActualTokenizerRefusal(unittest.TestCase):
 def make(self,path):
  t=Tokenizer(WordLevel({'[UNK]':0,'[EOS]':1,'[PAD]':2,'What':3,'is':4,'the':5,'capital':6,'of':7,'France':8,'?':9},unk_token='[UNK]'));t.pre_tokenizer=Whitespace()
  f=PreTrainedTokenizerFast(tokenizer_object=t,unk_token='[UNK]',eos_token='[EOS]',pad_token='[PAD]')
  f.chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}{% if add_generation_prompt %} assistant{% endif %}"
  f.save_pretrained(path);return f
 def test_real_chat_template_change_same_vocab_rejected_before_model(self):
  with tempfile.TemporaryDirectory()as d:
   a=Path(d)/'base';b=Path(d)/'checkpoint';self.make(a);other=self.make(b)
   other.chat_template="{{ messages[0]['content'] }} changed";other.save_pretrained(b)
   with self.assertRaisesRegex(ValueError,'chat_template differs'):m.hf_proof(a,b)
 def test_real_pad_change_same_vocab_rejected_before_model(self):
  with tempfile.TemporaryDirectory()as d:
   a=Path(d)/'base';b=Path(d)/'checkpoint';self.make(a);other=self.make(b)
   other.pad_token='[EOS]';other.save_pretrained(b)
   with self.assertRaisesRegex(ValueError,'pad/special tokens differ'):m.hf_proof(a,b)
class ActualAssetPathRefusal(unittest.TestCase):
 def test_locality_three_actual_paths_and_boundary_refusal(self):
  with tempfile.TemporaryDirectory()as d:
   parent=Path(d);root=parent/'new';root.mkdir();old=parent/'bapo/assets-root/assets';old.mkdir(parents=True)
   (old/'wiki').mkdir();(old/'e5-base-v2').mkdir();(old/'wiki/e5_Flat.index').write_text('index');(old/'wiki/wiki-18.jsonl').write_text('corpus')
   (root/'assets-root').mkdir();(root/'assets-root/assets').symlink_to(old,target_is_directory=True)
   run=root/'run';run.mkdir();raw=root/'raw';raw.mkdir()
   devices={str(i):{'uuid':'GPU-'+str(i)}for i in range(8)};totals={str(i):{'external_sm_upper_bound_percent':0,'external_memory_percent':0}for i in range(8)}
   (raw/'resource-watch.jsonl').write_text(json.dumps(dict(run=str(run),slurm_job_id='1',pause=False,time=1,allocated_gpus=devices,external_totals=totals))+'\n')
   (raw/'runtime-profile.txt').write_text('CUDA E5 GPU FAISS loopback127.0.0.1:18083')
   local=dict(proxy_env=[],host='h100-5',paths=[dict(path=str((old/name).resolve()),mount='/dev/a ext4 /dataset')for name in ('wiki/e5_Flat.index','wiki/wiki-18.jsonl','e5-base-v2')])
   path=raw/'retrieval-assets-locality.json'
   with patch.object(m,'ROOT',root):
    path.write_text(json.dumps(local));self.assertEqual(m.resource_proof(raw,run,1)['samples'],1)
    for kind in ('wronghost','empty','duplicate','wrongasset','nfs'):
     bad=copy.deepcopy(local)
     if kind=='wronghost':bad['host']='h100-50'
     if kind=='empty':bad['paths']=[]
     if kind=='duplicate':bad['paths'][1]=bad['paths'][0]
     if kind=='wrongasset':bad['paths'][0]['path']=str(parent/'arbitrary.index')
     if kind=='nfs':bad['paths'][0]['mount']='server:/a nfs /dataset'
     path.write_text(json.dumps(bad))
     with self.assertRaises(ValueError,msg=kind):m.resource_proof(raw,run,1)
if __name__=='__main__':unittest.main()
