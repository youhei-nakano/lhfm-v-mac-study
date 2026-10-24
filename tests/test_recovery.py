import json,hashlib,unittest
from pathlib import Path
from recovery_probe import replay
import torch
class RecoveryTests(unittest.TestCase):
 def test_cpu_save_resume(self):
  torch.set_num_threads(2);self.assertTrue(replay()['fixture_cpu_exact_model_ema_and_loss'])
 def test_replay_audit_and_frozen_conditions(self):
  r=json.loads(Path('recovery_replay_comparison.json').read_text());self.assertEqual(len(r['comparison']),10);self.assertTrue(all(x['loss_exact'] and x['grad_norm_exact'] for x in r['comparison']))
  f=json.loads(Path('RECOVERY_FREEZE.json').read_text())
  for name,digest in f['hashes'].items():self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),digest)
  self.assertEqual(json.loads(Path('configs/stage1.json').read_text())['steps'],3000)

 def test_transport_off_inactive_capacity_control(self):
  from architecture import Faithful
  torch.manual_seed(172);m=Faithful(mode='transport_off').eval();x=torch.rand(1,10,1,16,16)
  inactive=sum(p.numel() for p in m.transport.parameters())+sum(p.numel() for p in m.correction_transport.parameters())+m.history_out.weight[-32:].numel()+m.history_out.bias[-32:].numel()
  self.assertEqual(inactive,2160);self.assertEqual(sum(p.numel() for p in m.parameters())-inactive,63733)
  with torch.no_grad():
   a=m(x)
   for p in list(m.transport.parameters())+list(m.correction_transport.parameters()):p.add_(10)
   m.history_out.weight[-32:].add_(10);m.history_out.bias[-32:].add_(10);b=m(x)
  for left,right in zip(a,b):self.assertTrue(torch.equal(left,right))

 def test_partial_mac_convention(self):
  from arithmetic_scope import module_macs
  conv=torch.nn.Conv2d(1,2,3);x=torch.zeros(1,1,5,5);self.assertEqual(module_macs(conv,(x,),conv(x)),162)
  linear=torch.nn.Linear(4,3);y=torch.zeros(2,5,4);self.assertEqual(module_macs(linear,(y,),linear(y)),120)
