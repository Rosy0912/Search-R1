import unittest
from .archived_assets import resolve_assets, PJ_BASE, OLD_DATA, OLD_MODEL, PJ_MODEL, BUDGET_RUN
from .contracts import DATA_SHA


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


if __name__=='__main__':unittest.main()
