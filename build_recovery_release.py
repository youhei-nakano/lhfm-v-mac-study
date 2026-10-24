"""Lean faithful recovery release, no raw official data or environment."""
from pathlib import Path
import json,hashlib,zipfile
R=Path(__file__).resolve().parent
names=['make_recovery_report.py','arithmetic_scope.py','architecture.py','operators.py','dataset.py','metrics.py','runtime.py','stage1.py','recovery.py','recovery2.py','recovery_probe.py','recovery_probe787.py','summarize_recovery.py','summarize_recovery2.py','download_data.py','regenerate.py','smoke.py','build_recovery_release.py','audit_recovery_release.py','README_RECOVERY.md','RESULTS_RECOVERY.md','README.md','RESULTS.md','PLAN.md','RECOVERY_PLAN.md','RECOVERY_FREEZE.json','RECOVERY_CPU_PLAN.md','RECOVERY_CPU_FREEZE.json','recovery_probe787_cpu.json','BUDGET_AMENDMENT.json','FIDELITY_INHERITED.md','FIDELITY_STAGE1.md','recovery_device_audit.json','trajectory_prefix_audit.json','REFERENCE_MANIFEST.json','core_identity.json','configs/stage1.json','requirements-lock.txt','tests/test_formula.py','tests/test_stage_protocol.py','tests/test_recovery.py','tests.log','recovery_tests.log','data/test_source.json','recovery_probe_cpu.json','recovery_probe_mps.json','recovery_replay_comparison.json','recovery_delivery_scope.json','recovery_visual_review.json','runs/stage1/transport/latest_state.pt','runs/stage1/transport/best_ema.pt','runs/recovery1/transport/latest_state.pt']
files=[R/n for n in names if (R/n).is_file()]
for folder in ['runs/stage1','runs/recovery1','runs/recovery2']:
 for p in (R/folder).rglob('*'):
  if p.is_file() and (p.suffix in ['.json','.jsonl','.csv','.md','.png','.gif'] or p.name=='best_ema.pt' or p.name.startswith('example_') and p.suffix=='.npz'):files.append(p)
files=sorted(set(files),key=lambda p:str(p.relative_to(R)))
for p in files:
 assert '.venv' not in p.parts and '__pycache__' not in p.parts
 if p.suffix in ['.py','.md','.json','.jsonl','.csv','.log','.txt']:
  text=p.read_text();markers=['/'+s+'/' for s in ['Users','home']]+['gh'+'p_','github'+'_pat_','s'+'k-','AK'+'IA','-----BEGIN '+'PRIVATE KEY-----','Bearer'+' '];assert not any(s in text for s in markers),str(p.relative_to(R))
manifest={str(p.relative_to(R)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
archive=R/'lhfm-v-recovery-lean.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.writestr('lhfm-v-recovery/'+str(p.relative_to(R)),p.read_bytes())
 z.writestr('lhfm-v-recovery/MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
assert archive.stat().st_size<5*2**20,'Split before delivery if over5MiB'
meta={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files),'includes':'original576 full model/EMA/optimizer/RNG recovery state (small1.3MB), final selected EMA, all small logs, both attempt provenance','excludes':'official raw downloads, venv/cache, full prediction and dataset arrays, latest recovery optimizer'}
(R/'recovery_release_build.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
