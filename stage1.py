"""Frozen one-seed3000-update faithful scaled learning experiment, global2h cap."""
import argparse,copy,csv,hashlib,json,math,platform,time,os,datetime
from pathlib import Path
import numpy as np
import torch
from architecture import Faithful
from operators import objective,schedule
from dataset import digits,split,sequence
from metrics import evaluate
from runtime import memory,sync

SOURCES=['architecture.py','operators.py','dataset.py','metrics.py','runtime.py','stage1.py','PLAN.md','configs/stage1.json']
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
def cpu_tree(value):
 if torch.is_tensor(value):return value.detach().cpu().clone()
 if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
 if isinstance(value,list):return [cpu_tree(v) for v in value]
 if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
 return value
def source_hashes():return {name:digest(Path(name)) for name in SOURCES}

@torch.no_grad()
def predict(model,data,device,guard=None):
 values=[]
 for i,clip in enumerate(data):
  values.append(model(clip[:10][None].to(device))[0][0].cpu().numpy())
  if guard and (i%8==0 or i+1==len(data)):guard()
 return np.stack(values)

@torch.no_grad()
def validate(model,data,device,guard):
 p=predict(model,data,device,guard);return float(np.mean((p.astype(np.float64)-data[:,10:].numpy().astype(np.float64))**2))

def run(args):
 torch.set_num_threads(2);cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 device='mps' if torch.backends.mps.is_available() else 'cpu'
 if device!='mps':raise RuntimeError('MPS required for predeclared2h budget; no unmeasured CPU substitution')
 started=time.perf_counter();started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat();initial=memory(device);active={'phase':'prepare'}
 def event(record):
  record={'elapsed_seconds':time.perf_counter()-started,**record}
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
  print(json.dumps(record,allow_nan=False),flush=True)
 def guard():
  m=memory(device);elapsed=time.perf_counter()-started
  if m['peak_rss_bytes']>cfg['rss_limit_bytes']:raise RuntimeError('RSS limit')
  if m.get('driver_bytes',0)>cfg['mps_driver_limit_bytes']:raise RuntimeError('MPS driver limit')
  if m['swap_used_bytes']-initial['swap_used_bytes']>cfg['swap_growth_limit_bytes']:raise RuntimeError('Global swap growth limit')
  if elapsed>cfg['run_cap_seconds']:raise RuntimeError('Global7200second limit')
  return m
 hashes=source_hashes();dump(out/'source_hashes.json',hashes);dump(out/'config.json',cfg)
 dump(out/'runtime.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':device,'initial':initial,'pid':os.getpid(),'started_utc':started_utc,'initialization':'from scratch; no40update checkpoint loaded'})
 try:
  raw=digits();training_ids,val_ids=split();assert not set(training_ids)&set(val_ids)
  validation=torch.stack([sequence(raw,val_ids,i,cfg['validation_seed'])[0] for i in cfg['validation_ids']])
  np.savez_compressed(out/'validation.npz',frames=validation.numpy())
  dump(out/'data_manifest.json',{'training_digit_count':len(training_ids),'validation_digit_count':len(val_ids),'training_digit_indices_sha256':hashlib.sha256(training_ids.astype('<i8').tobytes()).hexdigest(),'validation_digit_indices_sha256':hashlib.sha256(val_ids.astype('<i8').tobytes()).hexdigest(),'validation_sequence_ids':cfg['validation_ids'],'reserved_official_test_ids':cfg['official_test_ids'],'test_inspected_before_training':False})
  event({'event':'prepared','seed':cfg['seed'],'steps_per_model':cfg['steps'],'validation_count':len(validation),'device':device,**guard()})
  for mode in cfg['models']:
   active={'phase':'training','model':mode,'update':0};folder=out/mode;folder.mkdir();torch.manual_seed(cfg['seed'])
   model=Faithful(mode=mode).to(device);ema=copy.deepcopy(model).eval();optimizer=torch.optim.AdamW(model.parameters(),lr=4e-5,betas=(.95,.999),eps=1e-8,weight_decay=1e-4)
   records=[];valid=[];best=float('inf');best_update=None;model_start=time.perf_counter();event({'event':'model_started','model':mode,'parameters':sum(p.numel() for p in model.parameters())})
   train_file=(folder/'train.csv').open('w',buffering=1);val_file=(folder/'validation.csv').open('w',buffering=1)
   train_writer=None;val_writer=csv.DictWriter(val_file,fieldnames=['update','ema_mse_raw','model_mse_raw','selected_best_update','ema_initial_weight']);val_writer.writeheader()
   for step in range(1,cfg['steps']+1):
    active['update']=step;tick=time.perf_counter();clip,_=sequence(raw,training_ids,step-1,cfg['training_data_seed']);x=clip[:10][None].to(device);target=clip[10:][None].to(device)
    lr,beta=schedule(step,cfg['steps'])
    for group in optimizer.param_groups:group['lr']=lr;group['betas']=(beta,.999)
    optimizer.zero_grad(set_to_none=True);p,e,t,c=model(x);loss,diag=objective(p,target,e,t,c)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
    with torch.no_grad():
     for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-cfg['ema_decay'])
    sync(device);record={'update':step,'loss':loss.item(),**{k:v.item() for k,v in diag.items()},'lr':lr,'beta1':beta,'grad_norm':norm.item(),'update_seconds':time.perf_counter()-tick,'ema_initial_weight':cfg['ema_decay']**step}
    if train_writer is None:train_writer=csv.DictWriter(train_file,fieldnames=record.keys());train_writer.writeheader()
    train_writer.writerow(record);records.append(record)
    dump(out/'current_status.json',{'pid':os.getpid(),'started_utc':started_utc,'model':mode,'completed_updates':step,'elapsed_seconds':time.perf_counter()-started,'phase':'training','initialization':'from scratch'})
    if step<=3 or step%25==0:event({'event':'update','model':mode,**record,**guard()})
    if step in cfg['validation_updates']:
     active['phase']='validation';score=validate(ema,validation,device,guard);raw_score=validate(model,validation,device,guard)
     if score<best:
      best=score;best_update=step
      torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},'mode':mode,'seed':cfg['seed'],'update':step,'validation_mse_raw':score,'source_hashes':hashes},folder/'best_ema.pt')
     vr={'update':step,'ema_mse_raw':score,'model_mse_raw':raw_score,'selected_best_update':best_update,'ema_initial_weight':cfg['ema_decay']**step};val_writer.writerow(vr);valid.append(vr)
     # Full rolling recovery state retained locally, never auto-resumed or silently restarted.
     torch.save({'model':cpu_tree(model.state_dict()),'ema':cpu_tree(ema.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),'cpu_rng_state':torch.get_rng_state(),'update':step,'mode':mode,'seed':cfg['seed'],'source_hashes':hashes},folder/'latest_state.pt')
     event({'event':'validation','model':mode,**vr,**guard()});active['phase']='training'
   train_file.close();val_file.close()
   torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},'mode':mode,'update':cfg['steps']},folder/'last_ema.pt')
   result={'model':mode,'seed':cfg['seed'],'updates':cfg['steps'],'best_update':best_update,'best_validation_mse_raw':best,'last_validation':valid[-1],'initial_ema_fraction_final':cfg['ema_decay']**cfg['steps'],'training_seconds_with_validation':time.perf_counter()-model_start,'memory':guard(),'parameters':sum(p.numel() for p in model.parameters()),'status':'completed'}
   dump(folder/'training_result.json',result);event({'event':'model_completed',**result});del model,ema,optimizer,p,e,t,c,loss;torch.mps.empty_cache()
  dump(out/'training_complete.json',{'models':2,'seconds':time.perf_counter()-started,'selected_by_validation_only':True})
  active={'phase':'test_source_verification'};test_path=Path(args.official_test);expected=json.loads(Path('data/test_source.json').read_text())['sha256'];assert digest(test_path)==expected
  assert source_hashes()==hashes
  # First opening/decoding the new heldout IDs occurs only after both selections are fixed.
  array=np.load(test_path,mmap_mode='r',allow_pickle=False);assert array.shape==(20,10000,64,64) and array.dtype==np.uint8
  test=torch.from_numpy(np.asarray(array[:,cfg['official_test_ids']]).transpose(1,0,2,3).copy()).float()[:,:,None]/255;np.savez_compressed(out/'official_test_subset.npz',frames=test.numpy());target=test[:,10:].numpy()
  test_results={}
  for mode in cfg['models']:
   active={'phase':'test','model':mode};folder=out/mode;saved=torch.load(folder/'best_ema.pt',weights_only=True,map_location='cpu');model=Faithful(mode=mode).to(device).eval();model.load_state_dict(saved['state_dict'])
   begin=time.perf_counter();prediction=predict(model,test,device,guard);sync(device);seconds=time.perf_counter()-begin;np.savez_compressed(folder/'prediction.npz',frames=prediction);scores=evaluate(prediction,target)
   context=test[:1,:10].to(device)
   with torch.no_grad():
    for _ in range(3):model(context)
    sync(device);timings=[]
    for _ in range(10):
     tick=time.perf_counter();model(context);sync(device);timings.append(time.perf_counter()-tick);guard()
   tr={'model':mode,'selected_update':saved['update'],'test_ids':cfg['official_test_ids'],'scores':scores,'inference_seconds_128_including_transfers':seconds,'latency':{'batch':1,'warmups':3,'repeats':10,'seconds':timings,'median_seconds':float(np.median(timings)),'scope':'64px10observed+10forecast model forward including computed-unused aux head and synchronize, excluding input transfer'},'memory':guard()}
   dump(folder/'test_result.json',tr);test_results[mode]=tr;event({'event':'test_completed','model':mode,'mse_spatial_sum_raw':scores['mse_spatial_sum_raw'],'mae_spatial_sum_raw':scores['mae_spatial_sum_raw'],'ssim':scores['ssim_range2'],'latency_median':tr['latency']['median_seconds'],**guard()})
   del model,prediction;torch.mps.empty_cache()
  dump(out/'last_frame.json',evaluate(test[:,9:10].repeat(1,10,1,1,1).numpy(),target));dump(out/'completed.json',{'status':'completed','seconds':time.perf_counter()-started,'test_opened_after_both_model_selection':True,'memory':guard()})
 except BaseException as error:
  event({'event':'stopped',**active,'type':type(error).__name__,'reason':str(error).replace(str(Path.cwd()),'<project>')});dump(out/'stopped.json',{'active':active,'type':type(error).__name__,'reason':str(error).replace(str(Path.cwd()),'<project>'),'elapsed_seconds':time.perf_counter()-started});raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/stage1.json');p.add_argument('--output',default='runs/stage1');p.add_argument('--official-test',default='data/mnist_test_seq.npy');run(p.parse_args())
