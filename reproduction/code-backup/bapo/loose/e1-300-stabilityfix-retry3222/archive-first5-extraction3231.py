"""Immutable extraction archive only; no checkpoint or receipt acceptance."""
import argparse,json,tarfile
from pathlib import Path
from first5_guard3231 import *
def inventory(directory):
    result=[]
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(),'Archive symlink refused')
        if path.is_dir():continue
        owned_file(path);result.append(dict(path=str(path.relative_to(directory)),bytes=path.stat().st_size,sha256=digest(path)))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--evidence-sha256',required=True);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    evidence=owned_file(a.evidence);require(evidence.name=='evidence.json'and digest(evidence)==a.evidence_sha256,'Exact extraction evidence required')
    e=read(evidence);evidence_identity(e)
    directory=evidence.parent;require(RUN/'logs'in directory.parents,'Run-local extraction directory required')
    tool=owned_file(Path(e['tool']['path']),ROOT);require(digest(tool)==EXTRACTOR_SHA,'Reviewed extractor changed')
    archive=a.archive;require(archive.is_absolute()and archive==archive.resolve()and archive.parent==RUN/'logs'and not archive.exists()and not archive.is_symlink(),'Fresh run/logs archive required')
    before=inventory(directory)
    with archive.open('xb')as f:
        with tarfile.open(fileobj=f,mode='w:gz')as tar:
            tar.add(directory,arcname=directory.name,recursive=False)
            for path in sorted(directory.rglob('*')):tar.add(path,arcname=str(Path(directory.name)/path.relative_to(directory)),recursive=False)
    require(inventory(directory)==before and digest(evidence)==a.evidence_sha256,'Extraction changed during archive; archive not accepted')
    print(json.dumps(dict(status='ARCHIVED_UNSIGNED_EXTRACTION',job_id=JOB,source=SOURCE,archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),evidence_sha256=a.evidence_sha256,files=before),allow_nan=False))
if __name__=='__main__':main()
