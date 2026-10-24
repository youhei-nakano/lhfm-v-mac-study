import json,math,unittest
from pathlib import Path

class StageProtocol(unittest.TestCase):
 def test_frozen_budget_and_unseen_test(self):
  c=json.loads(Path('configs/stage1.json').read_text())
  self.assertEqual(c['seed'],1017);self.assertEqual(c['steps'],3000);self.assertEqual(c['batch_size'],1)
  self.assertEqual(c['models'],['transport','transport_off']);self.assertEqual(c['official_test_ids'],list(range(16,144)))
  self.assertTrue(set(c['official_test_ids']).isdisjoint(range(16)));self.assertEqual(c['run_cap_seconds'],7200)
 def test_validation_milestone_rescaling_and_ema(self):
  c=json.loads(Path('configs/stage1.json').read_text())
  original=[5000,10000,20000]+list(range(40000,1240001,40000))+[1250000]
  self.assertEqual(c['validation_updates'],sorted(set(math.ceil(x*3000/1250000) for x in original)))
  self.assertEqual(c['ema_decay'],.999);self.assertAlmostEqual(.999**3000,.049712393998,places=10)

if __name__=='__main__':unittest.main()
