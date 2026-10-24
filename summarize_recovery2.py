"""Recompute the fixed stage1 results from saved arrays without further learning."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from metrics import evaluate
from recovery2 import source_hashes


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csv_rows(path):
 with path.open() as stream:return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
def close_dict(a,b):
 assert a.keys()==b.keys()
 for key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-10,atol=1e-10)

def summarize(root):
 cfg=json.loads((root/'config.json').read_text());complete=json.loads((root/'completed.json').read_text());assert complete['status']=='completed'
 assert json.loads((root/'source_hashes.json').read_text())==source_hashes()
 test=np.load(root/'official_test_subset.npz',allow_pickle=False)['frames'];assert test.shape==(128,20,1,64,64)
 assert cfg['official_test_ids']==list(range(16,144));target=test[:,10:];rows=[];summary={};training={};validation={};all_raw={};shards=[]
 for mode in cfg['models']:
  folder=root/mode;train=csv_rows(folder/'train.csv');val=csv_rows(folder/'validation.csv');result=json.loads((folder/'training_result.json').read_text());test_result=json.loads((folder/'test_result.json').read_text())
  assert len(train)==cfg['steps'] and [int(r['update']) for r in train]==list(range(1,3001))
  assert [int(r['update']) for r in val]==cfg['validation_updates']
  selected=min(val,key=lambda r:r['ema_mse_raw']);assert int(selected['update'])==result['best_update']==test_result['selected_update']
  prediction=np.load(folder/'prediction.npz',allow_pickle=False)['frames'];assert prediction.shape==(128,10,1,64,64)
  scores=evaluate(prediction,target);close_dict(scores,test_result['scores'])
  latency=test_result['latency'];np.testing.assert_allclose(np.median(latency['seconds']),latency['median_seconds'])
  for r in train:
   np.testing.assert_allclose(r['ema_initial_weight'],cfg['ema_decay']**int(r['update']),rtol=1e-12,atol=1e-12)
  summary[mode]={'selected_update':result['best_update'],'selected_ema_initial_fraction':cfg['ema_decay']**result['best_update'],'final_ema_initial_fraction':cfg['ema_decay']**cfg['steps'],'best_validation_mse_raw':result['best_validation_mse_raw'],'final_ema_validation_mse':val[-1]['ema_mse_raw'],'final_raw_model_validation_mse':val[-1]['model_mse_raw'],'scores':scores,'latency':latency,'training_seconds_with_validation':result['training_seconds_with_validation'],'inference_seconds_128_including_transfers':test_result['inference_seconds_128_including_transfers'],'memory':result['memory']}
  rows.append({'model':mode,'seed':cfg['seed'],'steps':cfg['steps'],'selected_update':result['best_update'],'selected_ema_initial_fraction':cfg['ema_decay']**result['best_update'],'mse_spatial_sum_raw':scores['mse_spatial_sum_raw'],'mae_spatial_sum_raw':scores['mae_spatial_sum_raw'],'ssim_range2':scores['ssim_range2'],'psnr_clipped':scores['psnr_clipped'],'latency_median_seconds':latency['median_seconds']})
  training[mode]=train;validation[mode]=val;all_raw[mode]=prediction
  for begin in range(0,128,32):
   name=f'raw_{mode}_{begin:03d}_{begin+31:03d}.npz';path=root/name
   np.savez_compressed(path,prediction=prediction[begin:begin+32],official_sequence_ids=np.asarray(cfg['official_test_ids'][begin:begin+32]))
   assert path.stat().st_size<8*2**20
   shards.append({'filename':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model':mode,'official_sequence_ids':cfg['official_test_ids'][begin:begin+32]})
 close_dict(evaluate(test[:,9:10].repeat(10,axis=1),target),json.loads((root/'last_frame.json').read_text()))
 summary['last_frame']=json.loads((root/'last_frame.json').read_text());summary['run']=complete
 summary['operational_recovery']={'original_aborted_attempt':'runs/stage1','resume_update':576,'original_aborted_updates':650,'discarded_from_final_trajectory':list(range(577,651)),'final_trajectory_rows_per_model':3000,'original_attempt_elapsed_seconds':json.loads(Path('runs/stage1/stopped.json').read_text())['elapsed_seconds'],'original_transport_prefix_seconds_including_validation':next(r['elapsed_seconds'] for r in map(json.loads,Path('runs/stage1/progress.jsonl').read_text().splitlines()) if r.get('event')=='validation' and r.get('update')==576),'replay_probe_in_final_trajectory':False,'same_mps_device':False,'device_note':'transport1..787MPS then788..3000CPU; off CPU throughout; all final evaluation/latency CPU','prior_recovery_attempt':'runs/recovery1','final_resume_update':787,'final_selected_trajectory':'original1..576 + recovery1 577..787 + recovery2 788..3000','prior_recovery_elapsed_seconds':json.loads(Path('runs/recovery1/stopped.json').read_text())['elapsed_seconds']}
 summary['comparison']={'mse_difference_transport_minus_off':summary['transport']['scores']['mse_spatial_sum_raw']-summary['transport_off']['scores']['mse_spatial_sum_raw'],'single_seed_only':True,'robust_superiority_claim':False}
 pairs=[]
 for i,sequence_id in enumerate(cfg['official_test_ids']):
  a=summary['transport']['scores']['per_sequence_mse'][i];b=summary['transport_off']['scores']['per_sequence_mse'][i];pairs.append({'official_sequence_id':sequence_id,'transport_mse_raw':a,'off_mse_raw':b,'difference_transport_minus_off':a-b})
 with (root/'paired_sequence_mse.csv').open('w') as stream:w=csv.DictWriter(stream,fieldnames=pairs[0]);w.writeheader();w.writerows(pairs)
 summary['comparison']['sequence_differences_scope']='128 paired sequences from one training seed; not128 independent training runs or a robustness/significance test'
 dump(root/'summary.json',summary);dump(root/'raw_prediction_manifest.json',{'scope':'All128 raw predictions,32sequence shards; not included in lean archive','shards':shards})
 with (root/'results.csv').open('w') as stream:w=csv.DictWriter(stream,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 dump(root/'verification.json',{'all_predictions_recomputed':True,'validation_selection_verified':True,'ema_state_verified':True,'latency_medians_verified':True,'core_and_plan_hashes_unchanged':True,'frozen_unseen_test_ids_16_143':True})
 # A tiny deterministic first heldout example for independent offline archive regeneration.
 np.savez_compressed(root/'example_official_16.npz',frames=test[:1])
 for mode in cfg['models']:np.savez_compressed(root/f'example_{mode}_16.npz',prediction=all_raw[mode][:1])
 figures(root,cfg,test,all_raw,training,validation)
 lines=['# Single-seed stage1 heldout comparison','','One initialization seed; no significance or robust superiority claim. Same128new official sequences after validation-only selection.','','| model | selected update | MSE sum raw | MAE sum raw | SSIM range2 | PSNR | latency ms |','|---|---:|---:|---:|---:|---:|---:|']
 for row in rows:lines.append('| '+row['model']+' | '+str(row['selected_update'])+' | '+' | '.join(f'{row[k]:.6f}' for k in ['mse_spatial_sum_raw','mae_spatial_sum_raw','ssim_range2','psnr_clipped'])+f" | {row['latency_median_seconds']*1000:.3f} |")
 lines+=['',f"Transport minus off MSE: {summary['comparison']['mse_difference_transport_minus_off']:.6f}"]
 (root/'TABLE.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

def figures(root,cfg,test,pred,training,validation):
 for index in range(4):
  rows={'Truth':test[index]}
  rows.update({m:np.concatenate((test[index,:10],p[index]),0) for m,p in pred.items()});rows['Last frame']=np.concatenate((test[index,:10],np.repeat(test[index,9:10],10,axis=0)),axis=0)
  im=Image.new('RGB',(1385,285),'white');d=ImageDraw.Draw(im);d.text((105,5),f'Observed0-9 / future10-19; fixed official sequence{cfg["official_test_ids"][index]}',fill='black')
  for r,(label,frames) in enumerate(rows.items()):
   d.text((2,40+r*64),label,fill='black')
   for t,frame in enumerate(frames):im.paste(Image.fromarray((frame[0].clip(0,1)*255).astype('uint8')).convert('RGB'),(105+t*64,25+r*64))
  im.save(root/f'official_sequence_{cfg["official_test_ids"][index]}.png')
 animation=[]
 for t in range(20):
  im=Image.new('RGB',(548,155),'white');d=ImageDraw.Draw(im)
  frames={'Truth':test[0]};frames.update({m:np.concatenate((test[0,:10],p[0]),0) for m,p in pred.items()});frames['Last frame']=np.concatenate((test[0,:10],np.repeat(test[0,9:10],10,axis=0)),axis=0)
  for i,(label,clip) in enumerate(frames.items()):
   d.text((i*137+2,3),label,fill='black');im.paste(Image.fromarray((clip[t,0].clip(0,1)*255).astype('uint8')).resize((128,128)).convert('RGB'),(i*137+2,23))
  animation.append(im)
 animation[0].save(root/'official_sequence16.gif',save_all=True,append_images=animation[1:],duration=200,loop=0)
 im=Image.new('RGB',(1100,420),'white');d=ImageDraw.Draw(im)
 colors={'transport':'red','transport_off':'blue'}
 for panel in range(2):
  x0=50+panel*550;y0=45;w=480;h=315;d.text((x0,10),['Training objective: trailing100update mean','Validation pixel MSE: solid EMA / dotted raw'][panel],fill='black')
  d.line((x0,y0,x0,y0+h,x0+w,y0+h),fill='black')
  arrays=[]
  for mode in cfg['models']:
   if panel==0:
    a=np.array([r['loss'] for r in training[mode]]);smoothed=np.array([a[max(0,i-99):i+1].mean() for i in range(len(a))]);arrays.append((mode,list(range(1,len(a)+1)),smoothed,'solid'))
   else:
    steps=[r['update'] for r in validation[mode]];arrays.append((mode,steps,[r['ema_mse_raw'] for r in validation[mode]],'solid'));arrays.append((mode,steps,[r['model_mse_raw'] for r in validation[mode]],'dots'))
  maximum=max(max(a[2]) for a in arrays)*1.05
  for mode,steps,values,style in arrays:
   points=[(x0+u/cfg['steps']*w,y0+h-v/maximum*h) for u,v in zip(steps,values)]
   if style=='solid':d.line(points,fill=colors[mode],width=2)
   else:
    for x,y in points:d.ellipse((x-2,y-2,x+2,y+2),fill=colors[mode])
  d.text((x0-35,y0),f'{maximum:.3f}',fill='black');d.text((x0+w-30,y0+h+10),'3000',fill='black')
 d.text((50,395),'Red transport / blue off. Raw model and EMA are separate; single initialization.',fill='black');im.save(root/'training_validation.png')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',type=Path,default=Path('runs/recovery2'));summarize(p.parse_args().directory)
