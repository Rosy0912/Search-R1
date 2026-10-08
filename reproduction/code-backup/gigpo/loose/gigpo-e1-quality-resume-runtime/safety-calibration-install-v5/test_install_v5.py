"""New local tar and no-overwrite checks; no remote, install or Slurm action."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest
import install_calibration_v5 as installer


def archive_fixture(tmp,mutation=None):
    target=tmp/'fixture.tar.gz';names=sorted(installer.MEMBERS)
    with tarfile.open(target,'w:gz') as archive:
        for i,name in enumerate(names):
            info=tarfile.TarInfo(name);payload=b'cpu-only-fixture';info.size=len(payload)
            if i==0:
                if mutation=='traversal':info.name='../native-v2/overwrite.py'
                elif mutation=='absolute':info.name='/native-v2/overwrite.py'
                elif mutation=='normalized':info.name=name.replace('/','//',1)
                elif mutation=='symlink':info.type=tarfile.SYMTYPE;info.linkname='/native-v2';info.size=0;payload=b''
                elif mutation=='hardlink':info.type=tarfile.LNKTYPE;info.linkname='/native-v2';info.size=0;payload=b''
                elif mutation=='directory':info.type=tarfile.DIRTYPE;info.size=0;payload=b''
                elif mutation=='duplicate':info.name=names[1]
                elif mutation=='oversized':payload=b'0'*(12*1024*1024+1);info.size=len(payload)
            archive.addfile(info,io.BytesIO(payload))
    return target,installer.sha_file(target)


@pytest.mark.parametrize('mutation',['traversal','absolute','normalized','symlink','hardlink','directory','duplicate','oversized'])
def test_tar_rejects_traversal_links_duplicates_nonfiles_and_oversize(tmp_path,mutation):
    archive,sha=archive_fixture(tmp_path,mutation)
    with pytest.raises(ValueError):installer.read_archive(archive,sha)


def test_archive_layer_allows_only_exact_regular_file_inventory(tmp_path):
    archive,sha=archive_fixture(tmp_path)
    assert set(installer.read_archive(archive,sha))==installer.MEMBERS
    with pytest.raises(ValueError,match='SHA'):installer.read_archive(archive,'f'*64)


def preflight_fixture(tmp,monkeypatch):
    root=tmp/'root';root.mkdir();(root/'tools').mkdir();(root/'logs').mkdir()
    monkeypatch.setattr(installer,'ROOT',root);monkeypatch.setattr(installer,'HELPERS',root/'tools/new-runtime')
    monkeypatch.setattr(installer,'CONTROLS',root/'tools/new-controls')
    monkeypatch.setattr(installer,'INTENT',root/'logs/intent');monkeypatch.setattr(installer,'RECEIPT',root/'logs/receipt')
    monkeypatch.setattr(installer.socket,'gethostname',lambda:'h100-8')
    original=root/'logs/proof';original.write_bytes(b'old')
    record=dict(path=str(original),bytes=3,sha256=installer.sha_file(original))
    return root,dict(install_files=[dict(record,existing_same_bytes_reuse_allowed=True)],required_existing_evidence=[])


@pytest.mark.parametrize('mutation',['existing_runtime','existing_control','existing_intent','existing_receipt','different_proof','symlink_proof','wrong_host'])
def test_existing_or_uncertain_targets_cannot_be_overwritten(tmp_path,monkeypatch,mutation):
    root,plan=preflight_fixture(tmp_path,monkeypatch)
    if mutation=='existing_runtime':installer.HELPERS.mkdir()
    elif mutation=='existing_control':installer.CONTROLS.mkdir()
    elif mutation=='existing_intent':installer.INTENT.write_text('claimed')
    elif mutation=='existing_receipt':installer.RECEIPT.write_text('uncertain')
    elif mutation=='different_proof':Path(plan['install_files'][0]['path']).write_bytes(b'new')
    elif mutation=='symlink_proof':
        path=Path(plan['install_files'][0]['path']);path.unlink();path.symlink_to(root/'elsewhere')
    else:monkeypatch.setattr(installer.socket,'gethostname',lambda:'h100-11')
    with pytest.raises(ValueError):installer.preflight(plan)


def test_identical_immutable_log_proof_may_be_reused_without_write(tmp_path,monkeypatch):
    root,plan=preflight_fixture(tmp_path,monkeypatch)
    path=Path(plan['install_files'][0]['path']);before=path.stat()
    assert installer.preflight(plan)==[str(path)]
    assert path.read_bytes()==b'old' and path.stat().st_mtime_ns==before.st_mtime_ns


def test_exclusive_writer_never_replaces_existing_file_or_symlink(tmp_path):
    path=tmp_path/'old';path.write_bytes(b'old')
    with pytest.raises(FileExistsError):installer.write_new(path,b'new')
    assert path.read_bytes()==b'old'
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(FileExistsError):installer.write_new(link,b'new')
    assert path.read_bytes()==b'old'


def test_actual_frozen_package_plan_is_null_native_and_separate_dirs():
    archive=Path(__file__).resolve().parent.with_name('safety-calibration-deployment-v5')/'policy-safety-calibration-52befe9-v5-package.tar.gz'
    plan=installer.installation_plan(installer.read_archive(archive))
    assert plan['runtime_helpers_count']==13 and plan['control_count']==5
    assert plan['transfer_ancillary_proofs']==15 and len(plan['required_existing_evidence'])==156
    assert not plan['ready'] and not plan['submitted'] and plan['source_native_acceptance'] is None
    assert all('/policy-safety-native-' not in r['path'] for r in plan['install_files'])
