"""Independent implementation of paper equations (24) and shifted series (0..24)."""
import math
import torch
from torch.nn import functional as F


def translate_zero(x,dy,dx):
    h,w=x.shape[-2:];out=F.pad(x,(2,2,2,2))
    return out[...,2-dy:2-dy+h,2-dx:2-dx+w]


def generator(image,velocity):
    """velocity channel0=x/column, channel1=y/row, pixels/frame (official convention)."""
    result=torch.zeros_like(image)
    for channel,(dy,dx) in enumerate([(0,1),(1,0)]):
        speed=velocity[:,channel:channel+1]
        for sign in [1,-1]:
            coefficient=(sign*speed).clamp_min(0)
            result=result+coefficient*(-1.5*image+2*translate_zero(image,sign*dy,sign*dx)-.5*translate_zero(image,2*sign*dy,2*sign*dx))
    return result


def exponential(image,source,velocity,dt=.5):
    if dt<=0 or not math.isfinite(dt):raise ValueError('positive finite dt')
    rate=float(velocity.detach().abs().sum(1).amax())
    if not math.isfinite(rate):raise FloatingPointError('nonfinite field')
    q=1.5*max(1.,rate);pieces=max(1,math.ceil(dt*q/3))
    if pieces>128:raise RuntimeError('Paper numerical work guard; no velocity clipping')
    mu=dt*q/pieces
    for _ in range(pieces):
        current=image;weight=math.exp(-mu);result=weight*current
        for order in range(1,25):
            current=current+generator(current,velocity)/q+source/q
            weight*=mu/order;result=result+weight*current
        image=result
    return image


def objective(prediction,target,source_energy,velocity_tv,coarse):
    b,t,c,h,w=target.shape
    pooled=F.avg_pool2d(target.detach().reshape(b*t,c,h,w),4,4).reshape(b,t,c,h//4,w//4)
    mse=(prediction-target).square().mean();plan=(coarse-pooled).square().mean()
    return mse+.001*source_energy+.0001*velocity_tv+.05*plan,{'mse':mse,'source_energy':source_energy,'velocity_tv':velocity_tv,'coarse_mse':plan}


def total_variation(w):return .5*((w[...,1:,:]-w[...,:-1,:]).abs().mean()+(w[...,:,1:]-w[...,:,:-1]).abs().mean())


def schedule(update,total):
    """Map shortened update indices to the original 1..1,250,000 schedule."""
    original=1+(update-1)*(1250000-1)/max(1,total-1)
    def base(index):
        step=index-1;boundary=.3*1250000-1
        if step<=boundary:
            fraction=step/boundary;a,b=4e-5,.001;c,d=.95,.85
        else:
            fraction=(step-boundary)/(1250000-1-boundary);a,b=.001,4e-9;c,d=.85,.95
        interpolate=lambda x,y:y+(x-y)*.5*(1+math.cos(math.pi*fraction))
        return interpolate(a,b),interpolate(c,d)
    lr,beta=base(original)
    if original>400000:
        initial=.5*base(400001)[0];fraction=(original-400001)/(1250000-400001)
        lr=4e-9+(initial-4e-9)*.5*(1+math.cos(math.pi*fraction))
    return lr,beta
