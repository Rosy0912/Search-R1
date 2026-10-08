"""Execute actual checker with isolated immutable files and mocked HTTP only."""
import pathlib,json,hashlib,sys,time,io
from unittest import mock
import pytest
SRC=pathlib.Path(__file__).parent/'check_h100_native_retriever.py'
@pytest.mark.parametrize('case',['positive','hash-change','evidence-change','permission','live-instance','expired','wrong-cluster','http-timeout','outside-root','missing-role','source-role-mismatch'])
def test_actual_native_checker_binding(tmp_path,case):
 root=tmp_path/'own';(root/'logs').mkdir(parents=True);evidence=root/'gate.json';evidence.write_text('reviewed gate raw');receipt=root/'logs/accepted.json';live={'cluster':'wuhan-h100','node':'h100-11','training_node':'h100-9','deadline_epoch':time.time()+8000,'gpu_uuid':'GPU-slurm','pid':123,'run':'own unique run'}
 if case=='expired':live['deadline_epoch']=time.time()+10
 if case=='wrong-cluster':live['cluster']='other'
 roles={}
 for role in ['baseline','gate','independent_review','service_source','guard_source','service_launcher_source','checker_source','training_launcher_source','probe_source']:
  file=root/(role+'.txt');file.write_text(role);roles[role]=str(file)
 hashes={str(pathlib.Path(f)):hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() for f in roles.values()}
 for key,role in [('source_sha256','service_source'),('guard_source_sha256','guard_source'),('launcher_source_sha256','service_launcher_source')]:live[key]=hashes[roles[role]]
 if case=='source-role-mismatch':live['source_sha256']='0'*64
 r={'status':'ACCEPTED_H100_GPU_RETRIEVAL_AFTER_INDEPENDENT_REVIEW','permitted_stages':['two'],'evidence_file_sha256':{str(evidence):hashlib.sha256(evidence.read_bytes()).hexdigest()},'instance':dict(live),'evidence_roles':roles}
 r['evidence_file_sha256'].update(hashes)
 if case=='missing-role':del r['evidence_roles']['probe_source']
 if case=='permission':r['permitted_stages']=['five']
 receipt.write_text(json.dumps(r));pin=hashlib.sha256(receipt.read_bytes()).hexdigest()
 if case=='hash-change':pin='0'*64
 if case=='evidence-change':evidence.write_text('polluted gate')
 if case=='live-instance':live['pid']=999
 if case=='outside-root':p=tmp_path/'elsewhere.json';p.write_bytes(receipt.read_bytes());receipt=p
 code=SRC.read_text().replace("'/dataset/shaoshitong/search-r1-rl/tree-grpo'",repr(str(root)))
 def urlopen(url,timeout):
  assert url=='http://10.32.0.11:18082/provenance' and timeout==10
  if case=='http-timeout':raise TimeoutError('actual bounded health failure fixture')
  return io.BytesIO(json.dumps(live).encode())
 def build(handler):
  assert handler.proxies=={}, 'Actual checker must explicitly suppress every environment proxy'
  return type('DirectOpener',(),{'open':staticmethod(urlopen)})()
 with mock.patch.object(sys,'argv',[str(SRC),'--receipt',str(receipt),'--sha256',pin,'--stage','two','--required-seconds','3600']),mock.patch('urllib.request.build_opener',side_effect=build),mock.patch.dict('os.environ',{'HTTP_PROXY':'http://forbidden-proxy.invalid:9','HTTPS_PROXY':'http://forbidden-proxy.invalid:9','http_proxy':'http://forbidden-proxy.invalid:9','https_proxy':'http://forbidden-proxy.invalid:9'}):
  if case=='positive':exec(compile(code,str(SRC),'exec'),{})
  else:
   with pytest.raises((AssertionError,TimeoutError)):exec(compile(code,str(SRC),'exec'),{})
