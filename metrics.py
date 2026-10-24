"""Paper evaluation conventions, independently expressed using standard SciPy."""
import numpy as np
from scipy.ndimage import uniform_filter

def evaluate(prediction,target):
    p=np.asarray(prediction);y=np.asarray(target)
    if p.shape!=y.shape or p.shape[1:]!=(10,1,64,64) or not np.isfinite(p).all():raise ValueError('prediction shape/finite')
    error=p.astype(np.float64)-y.astype(np.float64);clipped=p.clip(0,1).astype(np.float64);ce=clipped-y.astype(np.float64)
    mse=(error**2).mean((2,3,4));mae=abs(error).mean((2,3,4));cmse=(ce**2).mean((2,3,4))
    scores=np.empty(p.shape[:2],np.float64)
    for b in range(len(p)):
        for t in range(10):
            x=p[b,t].astype(np.float32).clip(0,1).swapaxes(0,2)[...,0];z=y[b,t].astype(np.float32).swapaxes(0,2)[...,0]
            ux=uniform_filter(x,7);uy=uniform_filter(z,7)
            vx=(uniform_filter(x*x,7)-ux*ux)*np.float32(49/48);vy=(uniform_filter(z*z,7)-uy*uy)*np.float32(49/48);cov=(uniform_filter(x*z,7)-ux*uy)*np.float32(49/48)
            s=((2*ux*uy+.02**2)*(2*cov+.06**2))/((ux*ux+uy*uy+.02**2)*(vx+vy+.06**2))
            scores[b,t]=np.float32(s[3:-3,3:-3].mean(dtype=np.float64))
    return {'mse_raw':float(mse.mean()),'mse_spatial_sum_raw':float(mse.mean()*4096),'mae_raw':float(mae.mean()),'mae_spatial_sum_raw':float(mae.mean()*4096),
            'ssim_range2':float(scores.mean()),'psnr_clipped':float((-10*np.log10(cmse.clip(1e-12))).mean()),
            'per_frame_mse_raw':mse.mean(0).tolist(),'per_sequence_mse':mse.mean(1).tolist(),'outside_0_1_fraction':float(((p<0)|(p>1)).mean())}
