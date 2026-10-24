"""Independent clean-tree CPU smoke, full components and operators, 16px fixture."""
import argparse,json
from pathlib import Path
import torch
from architecture import Faithful
from operators import objective
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(999)
 model=Faithful();x=torch.rand(1,10,1,16,16);target=torch.rand_like(x);opt=torch.optim.AdamW(model.parameters(),lr=4e-5,weight_decay=1e-4)
 prediction,e,t,c=model(x);loss,_=objective(prediction,target,e,t,c);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
 a.output.mkdir(parents=True,exist_ok=False);torch.save(model.state_dict(),a.output/'fixture.pt')
 (a.output/'result.json').write_text(json.dumps({'shape':list(prediction.shape),'loss':loss.item(),'gradient_finite':all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())})+'\n')
