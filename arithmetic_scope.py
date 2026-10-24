"""Auditable partial MACs, not total FLOPs; no timing claims from this instrumentation."""
import math,json,argparse
from pathlib import Path
import numpy as np
import torch
import architecture
from architecture import Faithful

def module_macs(module,inputs,output):
 if isinstance(module,torch.nn.Conv2d):return output.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size)
 if isinstance(module,torch.nn.Linear):return output.numel()*module.in_features
 return 0

def main(root):
 torch.set_num_threads(2);frames=torch.from_numpy(np.load(root/'example_official_16.npz',allow_pickle=False)['frames']);result={}
 for mode in ['transport','transport_off']:
  state=torch.load(root/mode/'best_ema.pt',map_location='cpu',weights_only=True);model=Faithful(mode=mode).eval();model.load_state_dict(state['state_dict']);counts={'Conv2d':0,'Linear':0};handles=[];pieces=[]
  def hook(module,inputs,output):counts[type(module).__name__]+=module_macs(module,inputs,output)
  for module in model.modules():
   if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):handles.append(module.register_forward_hook(hook))
  original=architecture.exponential
  def instrumented(image,source,velocity,dt=.5):
   rate=float(velocity.detach().abs().sum(1).amax());pieces.append(max(1,math.ceil(dt*1.5*max(1.,rate)/3)));return original(image,source,velocity,dt)
  architecture.exponential=instrumented
  try:
   with torch.no_grad():model(frames[:,:10])
  finally:
   architecture.exponential=original
   for h in handles:h.remove()
  total=sum(counts.values());result[mode]={'batch':1,'context':10,'forecast':10,'resolution':64,'counts':counts,'partial_conv_linear_macs':total,'partial_conv_linear_flops_2_per_mac':2*total,'exponential_half_steps':len(pieces),'internal_pieces_per_half':pieces,'generator_applications':24*sum(pieces),'parameters_total':sum(p.numel() for p in model.parameters()),'parameters_permanently_inactive_for_prediction_and_loss':2160 if mode=='transport_off' else 0}
 result['conventions']={'one_MAC':'one dot-product multiply-accumulate;2FLOPs by common convention;bias adds excluded','included':'Every executed Conv2d/Linear including recurrent repeated calls and aux/readouts (including unused off transport head)','excluded':'attention tensor products,normalization,activation,interpolation,elementwise operations,loss/TV and all physical-stencil/exponential arithmetic; data transfer/allocator/runtime not FLOPs','scope':'partial module MAC accounting, NOT full inference FLOPs or paper13.1G reproduction. Dynamic numerical work separately counted for fixed first heldout example; does not imply faster runtime.'}
 (root/'arithmetic_scope.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
