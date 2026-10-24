import time,resource,subprocess,re,platform
import torch

def sync(device):
    if device=='mps':torch.mps.synchronize()

def memory(device):
    text=subprocess.check_output(['sysctl','vm.swapusage'],text=True)
    swap=float(re.search(r'used = ([0-9.]+)M',text)[1])*2**20
    out={'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'swap_used_bytes':swap}
    if device=='mps':out.update(driver_bytes=torch.mps.driver_allocated_memory(),allocated_bytes=torch.mps.current_allocated_memory())
    return out

def guard(device,baseline,start,cap=1200):
    out=memory(device)
    if out['peak_rss_bytes']>1.5*2**30 or out.get('driver_bytes',0)>2**30:raise RuntimeError('Memory resource limit')
    if out['swap_used_bytes']-baseline>256*2**20:raise RuntimeError('Swap growth resource limit')
    if time.perf_counter()-start>cap:raise RuntimeError('Time resource limit')
    return out
