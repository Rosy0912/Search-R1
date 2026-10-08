import importlib.util,pathlib,json,hashlib,pytest
spec=importlib.util.spec_from_file_location('tested_headers',pathlib.Path(__file__).with_name('compiler_headers.py'));h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
def fixture(tmp):
 p=tmp/'overlay/usr/include/python3.10/Python.h';p.parent.mkdir(parents=True);p.write_bytes(b'official fixture')
 link=tmp/'overlay/usr/include/header-link';link.symlink_to('python3.10/Python.h')
 return {'files':{'usr/include/python3.10/Python.h':{'size':p.stat().st_size,'sha256':h.sha(p)}},'symlinks':{'usr/include/header-link':'python3.10/Python.h'}}
def test_actual_overlay_members_and_samepath_bytes_checked(tmp_path):
 inv=fixture(tmp_path);assert h.verify_tree(tmp_path,inv)['regular_files']==1
 (tmp_path/'overlay/usr/include/python3.10/Python.h').write_bytes(b'mutated')
 with pytest.raises(ValueError,match='inventory drift'):h.verify_tree(tmp_path,inv)
def test_added_or_changed_symlink_not_silently_ignored(tmp_path):
 inv=fixture(tmp_path);(tmp_path/'overlay/extra.pth').write_text('shadow')
 with pytest.raises(ValueError):h.verify_tree(tmp_path,inv)
 (tmp_path/'overlay/extra.pth').unlink();link=tmp_path/'overlay/usr/include/header-link';link.unlink();link.symlink_to('/unknown/global')
 with pytest.raises(ValueError):h.verify_tree(tmp_path,inv)
def test_static_pin_and_ray_worker_environment_actual_source():
 p=pathlib.Path(__file__).parent
 assert h.binding()['inventory_sha256']=='2fad02f3fdc7e74f7f734880022eb2e2b2edc94c24d597dd0b947d6a881a7201'
 assert "env['CPATH']=header_receipt['cpath']" in (p/'supervise.py').read_text()
 assert "('CPATH','PYTHONPATH'" in (p/'train_entry.py').read_text()
 assert 'verify_headers(config' in (p/'supervise.py').read_text()
