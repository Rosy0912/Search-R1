"""Original manifest actual bytes; tiny synthetic content-hash mechanics only."""
import hashlib,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import base_asset_ancestry as base

def test_original_source8e_manifest_exact_and_9files():
 records=base.original_base_records()
 assert len(records)==9
 assert {r['revision'] for r in records}=={base.MODEL_REVISION}
 assert [r['sha256'] for r in records if r['sha256']]==['f9558df91d3b89b4826e4db37439edb52f1d62a4fd602685013e7ca6b9f60f8f','51410930d5cf19a998fdb17ef0c46e4d9ace72c97a975a3331395a8a500f5edb']

@pytest.mark.parametrize('algorithm',['sha','blob'])
def test_rehashing_replacement_receipt_cannot_replace_original(tmp_path,algorithm):
 raw=b'original';p=tmp_path/'model';p.write_bytes(raw)
 record={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest() if algorithm=='sha' else None,'git_blob_sha1':hashlib.sha1(b'blob 8\0'+raw).hexdigest()}
 assert base.verify_original_record(p,record)==hashlib.sha256(raw).hexdigest()
 p.write_bytes(b'replaced')
 # A new receipt SHA is valid for current bytes; immutable origin still rejects.
 assert hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']
 with pytest.raises(ValueError,match='original'):base.verify_original_record(p,record)

def test_wrong_size_and_missing_original_file(tmp_path):
 p=tmp_path/'a';p.write_bytes(b'a')
 with pytest.raises(ValueError,match='size'):base.verify_original_record(p,{'size':2})
 with pytest.raises(FileNotFoundError):base.verify_original_record(tmp_path/'missing',{'size':1})

def test_extra_role_files_cannot_be_labelled_base(tmp_path):
 with pytest.raises(ValueError,match='nine'):base.verify_base_ancestry({'directory':str(tmp_path),'files':{'fake':'0'*64}})

def test_original_manifest_tamper_rejects(monkeypatch):
 monkeypatch.setattr(base,'TRAINING_ASSETS_SHA','0'*64)
 with pytest.raises(ValueError,match='manifest bytes'):base.original_base_records()
