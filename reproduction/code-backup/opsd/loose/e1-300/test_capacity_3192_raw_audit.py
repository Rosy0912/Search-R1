"""Pure CPU parse-boundary tests; no actual-job receipts are created."""
import importlib.util,pathlib,unittest
p=pathlib.Path(__file__).with_name('audit_capacity_3192_raw.py');s=importlib.util.spec_from_file_location('audit',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_aware_utc(self):self.assertEqual(m.utc('2026-10-04T12:00:00+00:00').utcoffset().total_seconds(),0)
 def test_non_utc(self):
  for text in ('now','2026-10-04T12:00:00','2026-10-04T20:00:00+08:00'):
   with self.assertRaises((ValueError,AssertionError)):m.utc(text)
 def test_duplicate_and_nonfinite(self):
  for text in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
   with self.assertRaises(ValueError):m.unique(text)
 def test_partial_ndjson_rejected(self):
  with self.assertRaises(ValueError):m.unique('{"time":"2026-10-04T12:00:00+00:00"')
if __name__=='__main__':unittest.main()
