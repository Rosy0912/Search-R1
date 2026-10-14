import hashlib,json,sys
from pathlib import Path

def main():
    root=Path(sys.argv[1]).resolve();manifest=json.loads(Path(sys.argv[2]).read_text())
    for name,expected in manifest['files'].items():
        p=root/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
    print('SOURCE_VERIFIED='+manifest['commit'],flush=True)
if __name__=='__main__':main()
