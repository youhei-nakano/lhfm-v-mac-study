"""Independently implement the official pinned generator protocol, no source import."""
import gzip,struct,hashlib
from pathlib import Path
import numpy as np
import torch
EXPECTED='440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'

def digits(path=Path('data/train-images-idx3-ubyte.gz')):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=EXPECTED:raise ValueError('MNIST training source hash')
    raw=gzip.decompress(data)
    if struct.unpack('>4I',raw[:16])!=(2051,60000,28,28) or len(raw)!=16+60000*784:raise ValueError('IDX header/size')
    return np.frombuffer(raw,np.uint8,offset=16).reshape(60000,28,28)

def split():
    order=np.random.default_rng(271100).permutation(60000)
    return order[5000:],order[:5000]

def sequence(raw,ids,sequence_id,seed=270829):
    rng=np.random.default_rng(np.random.SeedSequence([seed,sequence_id]))
    position=rng.uniform(0,1,(2,2));angles=rng.uniform(0,2*np.pi,2)
    velocity=.1*np.stack((np.cos(angles),np.sin(angles)),-1);selected=rng.choice(ids,2,replace=True)
    frames=np.zeros((20,64,64),np.uint8)
    for t in range(20):
        position+=velocity;hit=(position<=0)|(position>=1);velocity[hit]*=-1;position=np.clip(position,0,1)
        xy=np.floor(position*36).astype(np.int64)
        for digit,(x,y) in enumerate(xy):np.maximum(frames[t,y:y+28,x:x+28],raw[selected[digit]],out=frames[t,y:y+28,x:x+28])
    return torch.from_numpy(frames.copy()).float()[:,None]/255,selected.tolist()
