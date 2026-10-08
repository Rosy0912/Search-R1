import subprocess,types,pytest
from stream_transport import send,route

def test_actual_route_large_source_stays_stdin_short_argv():
 source='#'+('x'*300000)+'\nprint(1)\n';seen=[]
 def runner(argv,**kwargs):seen.append((argv,kwargs));return types.SimpleNamespace(returncode=0)
 send(source,runner=runner)
 argv,kwargs=seen[0];assert max(map(len,argv))<2048 and kwargs['input']==source.encode()
 assert 'python3 -' in argv[-1] and 'BatchMode=yes' in argv[-1] and 'LD_LIBRARY_PATH' in argv[-1]

def test_actual_remote_failure_not_retry_even_safe():
 seen=[]
 def runner(*a,**k):seen.append(1);return types.SimpleNamespace(returncode=1)
 assert send('print(1)',runner=runner,attempts=3,retry_safe=True,pause=lambda s:None).returncode==1 and len(seen)==1

def test_only_transport255_retry_has_same_source_and_route():
 seen=[]
 def runner(argv,**k):seen.append((argv,k['input']));return types.SimpleNamespace(returncode=255 if len(seen)==1 else 0)
 assert send('print(1)',runner=runner,attempts=3,retry_safe=True,pause=lambda s:None).returncode==0
 assert len(seen)==2 and seen[0]==seen[1]

def test_timeout_without_idempotent_retry_keeps_single_attempt():
 seen=[]
 def runner(*a,**k):seen.append(1);raise subprocess.TimeoutExpired('ssh',1)
 with pytest.raises(subprocess.TimeoutExpired):send('print(1)',runner=runner)
 assert seen==[1]

@pytest.mark.parametrize('source',[b'\x00binaryweights',b'\xff',b'not Python!'])
def test_non_source_rejected_before_ssh(source):
 with pytest.raises((ValueError,UnicodeError,SyntaxError)):send(source,runner=lambda *a,**k:pytest.fail('must not send'))
