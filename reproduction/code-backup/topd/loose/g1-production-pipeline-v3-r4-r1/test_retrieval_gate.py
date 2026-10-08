import pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import retrieval_gate as r
from checkpoint_inventory import sha

def fixture(tmp):
 names=list(r.REQUIRED)+[r.DIRECTORIES[0]+'/model.safetensors',r.DIRECTORIES[0]+'/tokenizer.json',r.DIRECTORIES[1]+'/faiss/gpu.so']
 files={}
 for n in names:
  p=tmp/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(n.encode());files[n]=sha(p)
 server=tmp/'repo/search_r1/search/retrieval_server.py';server.parent.mkdir(parents=True);server.write_text('source')
 return files,server,sha(server)

def test_actual_fresh_inventory_success_and_mutation_rejected(tmp_path):
 files,server,s=fixture(tmp_path)
 assert r.verify_inventory(tmp_path,files,server,s)['files'].keys()==files.keys()
 (tmp_path/next(iter(r.REQUIRED))).write_bytes(b'drift')
 with pytest.raises(ValueError,match='bytes drift'):r.verify_inventory(tmp_path,files,server,s)

def test_actual_extra_loader_file_rejected(tmp_path):
 files,server,s=fixture(tmp_path);(tmp_path/r.DIRECTORIES[0]/'added_config.json').write_text('{}')
 with pytest.raises(ValueError,match='exact file set'):r.verify_inventory(tmp_path,files,server,s)

def test_actual_server_and_symlink_rejected(tmp_path):
 files,server,s=fixture(tmp_path);server.write_text('changed')
 with pytest.raises(ValueError,match='server bytes'):r.verify_inventory(tmp_path,files,server,s)
 server.write_text('source');target=tmp_path/r.DIRECTORIES[1]/'alias';target.symlink_to(server)
 with pytest.raises(ValueError,match='symlink'):r.verify_inventory(tmp_path,files,server,s)
