import hashlib,json,pathlib,sys
import pytest
HERE=pathlib.Path(__file__).parent;sys.path.insert(0,str(HERE));import native_run

def fixture(tmp_path):
 overlay=tmp_path/'overlay';overlay.mkdir();(overlay/'module.py').write_bytes(b'official')
 r={'status':'COMPLETE_CPU_OVERLAY_ONLY','files_sha256':native_run.OVERLAY_FILES,'fixed_env_before':{'numpy':'same'},'fixed_env_after':{'numpy':'same'},'overlay':str(overlay),'overlay_files':{'module.py':native_run.sha(overlay/'module.py')}}
 receipt=tmp_path/'install-receipt.json';receipt.write_text(json.dumps(r));return receipt,r

def test_actual_complete_overlay_binding(tmp_path):
 receipt,r=fixture(tmp_path);assert native_run.verify_overlay(tmp_path,native_run.sha(receipt))[1]==tmp_path/'overlay'

def test_changed_overlay_bytes_are_rejected(tmp_path):
 receipt,r=fixture(tmp_path);(tmp_path/'overlay/module.py').write_bytes(b'changed')
 with pytest.raises(ValueError):native_run.verify_overlay(tmp_path,native_run.sha(receipt))

def test_failed_installer_cannot_start_fitting(tmp_path):
 receipt,r=fixture(tmp_path);r['status']='FAILED_PARTIAL_RETAINED';receipt.write_text(json.dumps(r))
 with pytest.raises(ValueError):native_run.verify_overlay(tmp_path,native_run.sha(receipt))
