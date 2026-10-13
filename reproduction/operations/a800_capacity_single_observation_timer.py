import argparse,json,os,sys,time,pathlib,subprocess,hashlib
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--not-before',type=float,required=True);p.add_argument('--read-source-sha256',required=True);a=p.parse_args();root=pathlib.Path(a.root);src=pathlib.Path(__file__).with_name('a800_owned_capacity_bounded_read.py');raw=src.read_bytes();assert hashlib.sha256(raw).hexdigest()==a.read_source_sha256;sys.path.insert(0,str(root/'actor-probes-ten'));from reproduction.a800_common.deploy import ssh_command
 print(json.dumps({'role':'single_local_capacity_waiter','pid':os.getpid(),'not_before_unix':a.not_before,'created_unix':time.time(),'remote_query_performed':False}),flush=True)
 while time.time()<a.not_before:time.sleep(min(30,a.not_before-time.time()))
 payload=json.loads((root/'preparation/a800-owned-free-training-gpu-reservation-single1-launch.json').read_text());payload['other_owned_reservation']=json.loads((root/'preparation/a800-owned-free-training-gpu-reservation-subset03-launch.json').read_text());r=subprocess.run(ssh_command(raw.decode()),input=json.dumps(payload),text=True,capture_output=True,timeout=90);now=time.time();out=root/'progress-checks'/('actor-probes-ten-resource-'+str(int(now))+'.json')
 if r.returncode==0:
  d=json.loads(r.stdout);d['local_monitor_recorded_unix']=now;d['extra_evaluation_or_training_started']=False
 else:d={'role':'bounded_capacity_observation_failed','observed_remote_state':'unknown','local_checked_unix':now,'returncode':r.returncode,'stderr_tail':r.stderr[-2000:],'no_training_or_reservation_retry':True}
 with out.open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'record':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'checked_unix':d.get('checked_unix',now),'remote_read_success':r.returncode==0,'own_ready':d.get('actual_owned_60GiB_each_verified',False) and d.get('other_actual_owned_60GiB_each_verified',False),'foreign_training_cuda_applications':d.get('foreign_training_cuda_applications'),'no_mutations_or_auto_retry':True}),flush=True)
if __name__=='__main__':main()
