import pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from retrieval_http_audit import inspect_log,require_healthy_log,RetrievalHTTPFailure

def test_actual_http_error_immediately_invalidates_original_silent_fallback(tmp_path):
 p=tmp_path/'retriever.log';p.write_text('INFO: 127.0.0.1:123 - "POST /retrieve HTTP/1.1" 200 OK\nINFO: 127.0.0.1:123 - "POST /retrieve HTTP/1.1" 500 Internal Server Error\n')
 captured=[]
 with pytest.raises(RetrievalHTTPFailure):require_healthy_log(p,captured.append)
 assert captured[0]['http_status_counts']=={'200':1,'500':1} and len(captured[0]['failures'])==1
@pytest.mark.parametrize('line',['ERROR:    Exception in ASGI application\n','Traceback (most recent call last):\n','INFO: 127.0.0.1 - "POST /retrieve HTTP/1.1" 422 Unprocessable Entity\n'])
def test_application_error_or_non200_cannot_be_a_pass(tmp_path,line):
 p=tmp_path/'r.log';p.write_text(line)
 with pytest.raises(RetrievalHTTPFailure):require_healthy_log(p)
def test_docs_and_non_error_trace_are_not_retrieval_failures(tmp_path):
 p=tmp_path/'r.log';p.write_text('INFO: 127.0.0.1 - "GET /docs HTTP/1.1" 200 OK\nINFO: 127.0.0.1 - "POST /retrieve HTTP/1.1" 200 OK\n')
 r=require_healthy_log(p);assert r['requests']==1 and r['failures']==[]

def test_missing_actual_service_log_cannot_be_health_pass(tmp_path):
 with pytest.raises(RetrievalHTTPFailure,match='missing'):require_healthy_log(tmp_path/'missing.log')
