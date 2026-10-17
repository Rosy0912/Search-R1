import unittest
from reproduction.seven_dataset.monitor_lumia import validated_rename_names

class RenameEvidenceTests(unittest.TestCase):
    def row(self):
        return dict(job='12', name='zhangjunming-ttt-eval-c2', before='JobId=12 JobName=old UserId=jmzhang(1066)', after='JobId=12 JobName=zhangjunming-ttt-eval-c2 UserId=jmzhang(1066)')
    def test_valid(self):
        self.assertEqual(validated_rename_names({'rows':[self.row()]})['12']['old'],'old')
    def test_wrong_id(self):
        r=self.row();r['after']=r['after'].replace('JobId=12','JobId=13')
        with self.assertRaises(AssertionError): validated_rename_names({'rows':[r]})
    def test_wrong_owner(self):
        r=self.row();r['after']=r['after'].replace('jmzhang','other')
        with self.assertRaises(AssertionError): validated_rename_names({'rows':[r]})
    def test_wrong_new_name(self):
        r=self.row();r['name']='unexpected'
        with self.assertRaises(AssertionError): validated_rename_names({'rows':[r]})
    def test_duplicate(self):
        with self.assertRaises(AssertionError): validated_rename_names({'rows':[self.row(),self.row()]})
