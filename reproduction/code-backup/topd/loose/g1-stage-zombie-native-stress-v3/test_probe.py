import pathlib,importlib.util,copy,select,pytest
s=importlib.util.spec_from_file_location('zombie_probe',pathlib.Path(__file__).with_name('probe.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def rows():
 live={'pid':10,'actual_pidfd_pid':10,'state':'S','proc_absent':False,'starttime_ticks':42,'uid':0,'poll_events':[],'cgroup':['1:freezer:/job','0::/system.slice']};dead=dict(live,state='Z',poll_events=[[9,select.POLLIN]],cgroup=['1:freezer:/','0::/system.slice']);return live,dead
def test_descriptor_proof_is_distinct_from_cgroup_observation():
 live,dead=rows();x=m.verify_exit(live,dead);assert x['cgroup_changed'] and x['held_pidfd_POLLIN']
@pytest.mark.parametrize('key,value',[('starttime_ticks',99),('uid',99),('state','S'),('poll_events',[]),('actual_pidfd_pid',-1)])
def test_live_drift_or_missing_held_death_evidence_refused(key,value):
 live,dead=rows();dead[key]=value
 with pytest.raises(ValueError):m.verify_exit(live,dead)

@pytest.mark.parametrize('cg',[['1:freezer:/foreign','0::/system.slice'],['1:freezer:/','0::/foreign']])
def test_unrelated_cgroup_migration_not_exit_proof(cg):
 live,dead=rows();dead['cgroup']=cg
 with pytest.raises(ValueError,match='cgroup migration'):m.verify_exit(live,dead)


def test_real_kernel_no_HUP_is_reported_not_invented():
 r={'proc_absent':True,'actual_pidfd_pid':-1,'poll_events':[[9,select.POLLIN|select.POLLRDNORM]]};x=m.verify_reaped([r]*5,0)
 assert x['actual_PID_minus_one_proc_absent_POLLIN_stable'] and x['actual_hangup_observed'] is False
@pytest.mark.parametrize('field,value',[('proc_absent',False),('actual_pidfd_pid',10),('poll_events',[])])
def test_postreap_unknown_or_live_evidence_cannot_pass(field,value):
 r={'proc_absent':True,'actual_pidfd_pid':-1,'poll_events':[[9,select.POLLIN]]};bad=dict(r);bad[field]=value
 with pytest.raises(ValueError):m.verify_reaped([r]*4+[bad],0)
