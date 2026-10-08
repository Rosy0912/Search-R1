import copy,unittest
from probe_round_checks import request_rows,validate_round_receipts,coexistence_evidence,resource_envelope

def records(step=1):
    ids=dict(run_id='hardware-fixture',code_commit='a'*40,config_sha256='b'*64,data_identity='c'*64,teacher_receipt_sha256='d'*64,source_sha256='e'*64)
    manifest=dict(ids,services=[dict(rank=i,pid=100+i,uuid=f'GPU-{i}') for i in range(8)],cgroup='/own/job',outer_updates=3)
    def state(device):return dict(parameter_sha256='f'*64,devices=[device],buffer_devices=[device],dtypes=['torch.float32'],training=False,training_modules_count=0,requires_grad_parameters=0,grad_parameters=0)
    receipts=[]
    for i in range(8):
        gpu=dict(pid=100+i,uuid=f'GPU-{i}',used_mib=13000)
        receipts.append(dict(ids,rank=i,pid=100+i,uuid=f'GPU-{i}',cgroup='/own/job',outer_step=step,rows=2,status='RELEASED',allocator_released=dict(allocated_bytes=0,reserved_bytes=0),targets_none=True,dtype='float32',tf32=False,attention='eager',gpu_pid_phase_sample=gpu,gpu_pid_released=gpu,loaded_cpu=state('cpu'),released_cpu=state('cpu'),gpu_before=state('cuda:0') if step==1 else None,gpu_after=state('cuda:0') if step==3 else None,scored_tokens_delta=6432 if i==7 and step==3 else 100))
    return receipts,manifest

class Checks(unittest.TestCase):
    def test_valid_three_rounds(self):
        for step in (1,2,3):validate_round_receipts(*records(step),step)
    def test_unique_row_order(self):
        p,r,a=request_rows([11,12],0,1);self.assertEqual(len(set(map(tuple,p))),16);self.assertEqual(len(set(map(tuple,r))),16)
        self.assertTrue(all(len(row)==256 for row in p));self.assertTrue(all(sum(row)==32 for row in a))
    def test_longest_context_no_truncation(self):
        p,r,a=request_rows([11,12],0,3);self.assertTrue(all(len(row)==2048 for row in p));self.assertTrue(all(len(row)==4096 for row in r));self.assertEqual(sum(v!=0 for v in p[15])+sum(a[15]),6144);self.assertEqual(sum(a[14]),32);self.assertEqual(sum(a[13]),500)
    def test_invalid_input(self):
        for tokens,pad,step in [([11],0,1),([11,12],11,1),([11,12],0,4)]:
            with self.assertRaises(ValueError):request_rows(tokens,pad,step)
    def test_workspace_not_waived(self):
        r,m=records();r[0]['allocator_released']['allocated_bytes']=33554432
        with self.assertRaises(ValueError):validate_round_receipts(r,m,1)
    def test_frozen_parameter_changed(self):
        r,m=records();r[0]['released_cpu']['parameter_sha256']='bad'
        with self.assertRaises(ValueError):validate_round_receipts(r,m,1)
    def test_initial_gpu_snapshot_missing(self):
        r,m=records();r[0]['gpu_before']=None
        with self.assertRaises(ValueError):validate_round_receipts(r,m,1)
    def test_last_gpu_snapshot_missing(self):
        r,m=records(3);r[0]['gpu_after']=None
        with self.assertRaises(ValueError):validate_round_receipts(r,m,3)
    def test_truncated_long_context(self):
        r,m=records(3);r[7]['scored_tokens_delta']=6400
        with self.assertRaises(ValueError):validate_round_receipts(r,m,3)
    def test_foreign_peer_identity(self):
        for key,value in [('pid',999),('uuid','GPU-foreign'),('cgroup','/foreign'),('source_sha256','bad')]:
            r,m=records();r[0][key]=value
            with self.assertRaises(ValueError):validate_round_receipts(r,m,1)
    def test_cpu_device_and_buffers(self):
        for key in ('devices','buffer_devices'):
            r,m=records();r[0]['released_cpu'][key]=['cuda:0']
            with self.assertRaises(ValueError):validate_round_receipts(r,m,1)
    def test_math_gates(self):
        for key,value in [('dtype','bfloat16'),('tf32',True),('targets_none',False),('attention','flash')]:
            r,m=records();r[0][key]=value
            with self.assertRaises(ValueError):validate_round_receipts(r,m,1)
    def test_sampled_rss_parent_and_shared_scope(self):
        parent=dict(pid=1,memory=['VmRSS: 100 kB','VmHWM: 200 kB'],user_cpu_ticks=3,system_cpu_ticks=4)
        child=dict(pid=2,memory=['VmRSS: 150 kB','VmHWM: 250 kB'],user_cpu_ticks=5,system_cpu_ticks=6)
        e=resource_envelope([child,parent,dict(pid=3,exited=True)],parent)
        self.assertEqual(e['summed_rss_kib'],250);self.assertEqual(e['summed_per_pid_hwm_kib'],450);self.assertEqual(e['alive_sampled_pids'],[1,2]);self.assertEqual(e['user_cpu_ticks'],8);self.assertIn('shared pages',e['scope'])
    def test_rss_units_fail_closed(self):
        parent=dict(pid=1,memory=['VmRSS: 100 MB'],user_cpu_ticks=0,system_cpu_ticks=0)
        with self.assertRaises(ValueError):resource_envelope([],parent)
    def test_actual_full_coexistence(self):
        _,m=records();mapping=[dict(uuid=s['uuid'],index=i) for i,s in enumerate(m['services'])];pids=list(range(200,208))
        lines=[f'GPU-{i}, {100+i}, 13000' for i in range(8)]+[f'GPU-{i}, {200+i}, 46000' for i in range(8)]+[f'GPU-{i}, 300, 7000' for i in range(8)]
        sample=dict(time=1,phase='scoring',processes=lines)
        self.assertEqual(len(coexistence_evidence([sample],m['services'],pids,mapping,300)),1)
        for change in ('scorer','vllm','retriever'):
            bad=copy.deepcopy(sample);idx={'scorer':0,'vllm':8,'retriever':16}[change];bad['processes'].pop(idx)
            self.assertEqual(coexistence_evidence([bad],m['services'],pids,mapping,300),[])
        sample['processes'][0]='GPU-0, 100, 500';self.assertEqual(coexistence_evidence([sample],m['services'],pids,mapping,300),[])

if __name__=='__main__':unittest.main()
