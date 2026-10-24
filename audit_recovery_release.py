"""Clean extraction, complete manifest, formula/recovery tests, fixture and regeneration."""
import json,hashlib,zipfile,tempfile,subprocess,sys,os
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;archive=root/'lhfm-v-recovery-lean.zip';report={'archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'official_source_executed':False}
with tempfile.TemporaryDirectory(prefix='lhfm-recovery-audit-') as temporary:
 base=Path(temporary)
 with zipfile.ZipFile(archive) as z:
  for name in z.namelist():assert (base/name).resolve().is_relative_to(base.resolve())
  z.extractall(base)
 project=base/'lhfm-v-recovery';manifest=json.loads((project/'MANIFEST.json').read_text())
 for name,m in manifest.items():
  data=(project/name).read_bytes();assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256']
 report['manifest_files']=len(manifest);env=dict(os.environ);env.pop('PYTHONPATH',None)
 def command(args,timeout=180):
  r=subprocess.run([sys.executable,*args],cwd=project,env=env,capture_output=True,text=True,timeout=timeout)
  if r.returncode:raise RuntimeError('Audit failure '+args[0]+': '+r.stderr[-1500:].replace(temporary,'<temporary>'))
  return r
 tests=command(['-m','unittest','discover','-s','tests','-v']);assert 'Ran 14 tests' in tests.stderr and tests.stderr.rstrip().endswith('OK');report['tests']='14 passed'
 command(['-m','pip','check']);command(['smoke.py','--output','audit-smoke']);report['full_component_cpu_smoke']=True
 scope=json.loads((project/'recovery_delivery_scope.json').read_text());differences={}
 for mode in scope['models_available']:
  name=f'audit_{mode}.npz';command(['regenerate.py',f"{scope.get('run_root','runs/recovery2')}/{mode}/best_ema.pt",scope['example_file'],'--output',name])
  generated=np.load(project/name)['prediction'];saved=np.load(project/scope['example_prediction_files'][mode])['prediction'];difference=float(np.max(abs(generated-saved)));np.testing.assert_allclose(generated,saved,rtol=3e-4,atol=3e-4);differences[mode]=difference
 report['cpu_regeneration_max_abs_vs_saved_example']=differences;report['experiment_status']=scope['status'];report['status']='passed'
(root/'recovery_release_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
