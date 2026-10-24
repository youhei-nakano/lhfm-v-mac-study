"""Operational recovery tests; no heldout data or research-source execution."""
import copy,time,json,gc,sys,hashlib
from pathlib import Path
import torch
from architecture import Faithful
from operators import objective,schedule
from dataset import digits,split,sequence
from stage1 import cpu_tree
from runtime import memory,sync

def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=4e-5,betas=(.95,.999),eps=1e-8,weight_decay=1e-4)
def update(model,ema,opt,clip,step,device):
 x=clip[:10][None].to(device);target=clip[10:][None].to(device);lr,beta=schedule(step,3000)
 for group in opt.param_groups:group['lr']=lr;group['betas']=(beta,.999)
 opt.zero_grad(set_to_none=True);p,e,t,c=model(x);loss,diag=objective(p,target,e,t,c);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
 with torch.no_grad():
  for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.001)
 sync(device);result={'loss':loss.item(),'lr':lr,'beta1':beta,'grad_norm':norm.item()}
 del x,target,p,e,t,c,loss,diag,norm;opt.zero_grad(set_to_none=True);gc.collect()
 if device=='mps':torch.mps.empty_cache()
 return result

def replay():
 torch.manual_seed(123);a=Faithful();ea=copy.deepcopy(a);oa=optimizer(a);clip=torch.rand(20,1,16,16)
 update(a,ea,oa,clip,1,'cpu');state=cpu_tree({'model':a.state_dict(),'ema':ea.state_dict(),'optimizer':oa.state_dict(),'rng':torch.get_rng_state()})
 expected=update(a,ea,oa,clip,2,'cpu');b=Faithful();eb=copy.deepcopy(b);ob=optimizer(b);b.load_state_dict(state['model']);eb.load_state_dict(state['ema']);ob.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);actual=update(b,eb,ob,clip,2,'cpu')
 for left,right in [(a,b),(ea,eb)]:
  for x,y in zip(left.parameters(),right.parameters()):assert torch.equal(x,y)
 assert expected==actual
 return {'fixture_cpu_exact_model_ema_and_loss':True,'schedule':'pure function of absolute update, no separate scheduler state','data':'per-sequence SeedSequence(training_seed,absolute_update-1), no mutable dataset RNG','training_stochastic_ops':'no dropout or stochastic forward operations; initial CPU RNG saved'}
if __name__=='__main__':
 torch.set_num_threads(2);device=sys.argv[1] if len(sys.argv)>1 else 'cpu';initial=memory(device)
 def protected():
  m=memory(device)
  if m['peak_rss_bytes']>1.5*2**30 or m.get('driver_bytes',0)>2**30 or m['swap_used_bytes']-initial['swap_used_bytes']>256*2**20:raise RuntimeError('Unchanged recovery probe resource guard')
  return m
 protected();r={'replay':replay(),'device':device};saved=torch.load('runs/stage1/transport/latest_state.pt',weights_only=True,map_location='cpu');a=Faithful().to(device);ea=copy.deepcopy(a);oa=optimizer(a);a.load_state_dict(saved['model']);ea.load_state_dict(saved['ema']);oa.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['cpu_rng_state']);raw=digits();ids,_=split();r['before']=protected();r['updates']=[]
 for step in range(577,587):
  clip,_=sequence(raw,ids,step-1);begin=time.perf_counter();v=update(a,ea,oa,clip,step,device);r['updates'].append({'update':step,'seconds':time.perf_counter()-begin,**v,**protected()})
 Path('recovery_probe_'+device+'.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
