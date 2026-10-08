import importlib.util,json,pathlib,sys,types
import pytest
HERE=pathlib.Path(__file__).parent
sys.path.insert(0,str(HERE));import install

def test_official_records_strict_four_compatible_packages():
 r=install.records();assert len(r)==4;assert {x['package']:x['version'] for x in r}==install.PACKAGES
 assert sum(x['official_size'] for x in r)==54846000

def test_full_wheel_sha_and_size_are_both_required(tmp_path):
 p=tmp_path/'wheel';p.write_bytes(b'official');r={'official_size':8,'official_sha256':install.sha(p)}
 install.verify(p,r);p.write_bytes(b'changed!')
 with pytest.raises(ValueError):install.verify(p,r)

def test_symlink_wheel_rejected(tmp_path):
 p=tmp_path/'wheel';p.write_bytes(b'official');alias=tmp_path/'alias';alias.symlink_to(p)
 with pytest.raises(ValueError):install.verify(alias,{'official_size':8,'official_sha256':install.sha(p)})

def test_install_cli_has_no_deps_no_index_only_target_no_compile(tmp_path):
 c=install.pip_command([tmp_path/'wheel.whl'],tmp_path/'overlay')
 assert c[0]==str(install.ROOT/'tools/uv');assert '--no-deps' in c and '--no-index' in c and '--offline' in c and '--no-cache' in c and '--no-config' in c and '--no-python-downloads' in c
 assert '--compile-bytecode' not in c and c[c.index('--python')+1]==sys.executable
 assert c[c.index('--target')+1]==str(tmp_path/'overlay');assert '--upgrade' not in c

def test_interrupted_official_wheel_resumes_preserving_exact_offset(tmp_path,monkeypatch):
 import hashlib
 class NetworkError(Exception):pass
 calls=[]
 class Response:
  def __init__(self,n):self.n=n;self.status_code=200 if n==0 else 206;self.headers={} if n==0 else {'Content-Range':'bytes 3-7/8'}
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def raise_for_status(self):pass
  def iter_content(self,n):
   if self.n==0:yield b'off';raise NetworkError('fixture interrupted body')
   yield b'icial'
 class Session:
  trust_env=False
  def get(self,url,**kwargs):calls.append(kwargs);return Response(len(calls)-1)
  def close(self):pass
 monkeypatch.setitem(sys.modules,'requests',types.SimpleNamespace(Session=Session,RequestException=NetworkError))
 monkeypatch.setattr(install.time,'sleep',lambda duration:None)
 item={'filename':'fixture.whl','url':'https://files.pythonhosted.org/fixture.whl','official_size':8,'official_sha256':hashlib.sha256(b'official').hexdigest()}
 result=install.download(item,tmp_path)
 assert (tmp_path/'fixture.whl').read_bytes()==b'official';assert calls[1]['headers']=={'Range':'bytes=3-'}
 assert result['events'][0]['bytes_after']==3 and result['target_sha256']==item['official_sha256']
