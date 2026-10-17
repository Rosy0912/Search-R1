import unittest
from .archived_assets import resolve_assets, PJ_BASE, OLD_DATA, OLD_MODEL, PJ_MODEL, BUDGET_RUN
from .contracts import DATA_SHA, GROUPS, milestone_job_name


class ArchivedAssetsTest(unittest.TestCase):
    def record(self, **changes):
        return dict(dict(group='budget-5',run_id=BUDGET_RUN,data_identity=DATA_SHA,
                         data=OLD_DATA,model=OLD_MODEL),**changes)

    def test_exact_migration_preserves_receipt(self):
        receipt=self.record();before=dict(receipt)
        assets,binding=resolve_assets(PJ_BASE,'budget-5','pj_pt',receipt)
        self.assertEqual(receipt,before)
        self.assertEqual(str(assets['data']),PJ_BASE+'/common-data')
        self.assertEqual(str(assets['model']),PJ_MODEL)
        self.assertEqual(binding['model']['archived'],OLD_MODEL)

    def test_already_physical_has_no_alias(self):
        _,binding=resolve_assets(PJ_BASE,'budget-5','pj_pt',self.record(data=PJ_BASE+'/common-data',model=PJ_MODEL))
        self.assertEqual(binding,{})

    def test_lumia_not_rewritten(self):
        assets,binding=resolve_assets('/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1','budget-5','slurm_lumia',self.record())
        self.assertEqual(str(assets['model']),OLD_MODEL);self.assertEqual(binding,{})

    def test_unknown_asset_rejected(self):
        for changes in ({'data':OLD_DATA+'/../other'},{'model':OLD_MODEL+'-other'}):
            with self.assertRaises(ValueError):resolve_assets(PJ_BASE,'budget-5','pj_pt',self.record(**changes))

    def test_wrong_identity_rejected(self):
        for changes in ({'run_id':'different'},{'data_identity':'different'},{'group':'generation-2'}):
            with self.assertRaises(ValueError):resolve_assets(PJ_BASE,'budget-5','pj_pt',self.record(**changes))

    def test_unknown_host_base_rejected(self):
        with self.assertRaises(ValueError):resolve_assets(PJ_BASE+'-other','budget-5','pj_pt',self.record())

    def test_pt_names_fit_actual_api_limit_and_are_unique(self):
        names=[]
        for group in GROUPS:
            for step in ([50] if group.startswith('position-') else [50,100,150]):
                name=milestone_job_name(group,step,'a'*64,'pj_pt')
                self.assertLessEqual(len(name),32);names.append(name)
        self.assertEqual(len(names),len(set(names)))

    def test_lumia_names_unchanged(self):
        self.assertEqual(milestone_job_name('C2-random-final',150,'a'*64,'slurm_lumia'),
                         'rt16-seven-c2-random-final-150-aaaaaaaa')


if __name__=='__main__':unittest.main()
