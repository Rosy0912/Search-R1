"""CPU invented receipts test rejection rules, never real native execution."""
import copy,json,math,unittest
from reproduction.common8.formal_contract import consumed_lr
from reproduction.common8.safety_calibration_evidence import EvidenceError
from reproduction.common8.test_safety_calibration_evidence import fabricated_segment,digest
from reproduction.common8.warmup_measurement_evidence import verify_segment
from reproduction.common8.warmup_measurement_plan import ROLE


def two_outers():
    logs,binding=fabricated_segment(2,role=ROLE);binding['schema_version']=2
    for rows in logs.values():
        for row in rows:
            if row['kind']=='OUTER_BUDGET':continue
            step=row['context']['logical_update'];row['context']['behavior_probability_gate_sha256']=digest('CPU gate'+str(step))
            if row['kind']=='OPTIMIZER_WRITE':
                row['learning_rates']=[consumed_lr(step)];continue
            row['enforcement']['negative_tail']='observe_only'
            # An independent constant0.01 log-ratio has complete small positive KL.
            row.update(sampled_old_policy_kl=math.expm1(.01)-.01,
                weighted_k3_numerator=row['effective_weight_mass']*(math.expm1(.01)-.01),
                tail_effective_weight_mass=0.,negative_tail_count=0,
                negative_tail_count_fraction=0.,negative_tail_weight_fraction=0.,
                dual_clip_saturated_objective_weight_fraction=0.,
                negative_log_ratio_p95=.01,negative_log_ratio_p99=.01,negative_log_ratio_max=.01)
    return logs,binding


class WarmupMeasurementEvidenceTests(unittest.TestCase):
    def test_two_outers_deduplicate640_rank_checks_to80_global_and_roundtrip(self):
        logs,binding=two_outers();original=copy.deepcopy(logs)
        result=verify_segment(logs,binding)
        self.assertEqual(result['optimizer_writes_per_rank'],40)
        self.assertEqual(len(result['global_monitors']),80)
        self.assertEqual(set(result['behavior_gate_sha256']),{'1','2'})
        self.assertEqual(json.loads(json.dumps(result)),result)
        self.assertEqual(logs,original);self.assertFalse(result['production_admission'])

    def test_wrong_constant_lr_or_skipped_warmup_rejected(self):
        for index,value in ((1,1e-6),(62,0.),(62,1e-6)):
            logs,binding=two_outers();logs[0][index]['learning_rates']=[value]
            with self.subTest(index=index,value=value),self.assertRaises(EvidenceError):verify_segment(logs,binding)

    def test_missing_duplicate_reordered_event_or_wrong_second_gate_rejected(self):
        for change in ('missing','duplicate','reorder','gate'):
            logs,binding=two_outers()
            if change=='missing':logs[7].pop(62)
            elif change=='duplicate':logs[7].append(copy.deepcopy(logs[7][62]))
            elif change=='reorder':logs[7][61],logs[7][62]=logs[7][62],logs[7][61]
            else:logs[7][62]['context']['behavior_probability_gate_sha256']=digest('foreign gate')
            with self.subTest(change=change),self.assertRaises(EvidenceError):verify_segment(logs,binding)

    def test_kl_brake_stays_active_and_postclip_bound_stays_one(self):
        for change in ('KL','clipping','negative_tail'):
            logs,binding=two_outers()
            if change=='KL':
                for rows in logs.values():
                    rows[61]['sampled_old_policy_kl']=.02
                    rows[61]['weighted_k3_numerator']=rows[61]['effective_weight_mass']*.02
            elif change=='clipping':logs[7][62]['gradient_norm_after']=1.01
            else:logs[7][61]['enforcement']['negative_tail']='enforced'
            with self.subTest(change=change),self.assertRaises(EvidenceError):verify_segment(logs,binding)

    def test_old_schema_role_and_wrong_global_budget_never_promoted(self):
        for key,value in (('schema_version',1),('role','policy_safety_calibration_discovery'),('outer_count',1)):
            logs,binding=two_outers();binding[key]=value
            with self.subTest(key=key),self.assertRaises(EvidenceError):verify_segment(logs,binding)
        logs,binding=two_outers();logs[7][-1]['cumulative_writes']=20
        with self.assertRaises(EvidenceError):verify_segment(logs,binding)


if __name__=='__main__':unittest.main()
