"""Independent semantic reimplementation from pinned, read-only architecture reference.

NHWC recurrent states, NCHW convolutions. Widths may scale; component semantics stay.
No official source is imported or executed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from operators import exponential,total_variation


def norm(c):
    groups=min(8,c)
    while c%groups:groups-=1
    return nn.GroupNorm(groups,c)

class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.n1=norm(c);self.n2=norm(c);self.a=nn.Conv2d(c,c,3,padding=1);self.b=nn.Conv2d(c,c,3,padding=1)
    def forward(self,x):return x+self.b(F.silu(self.n2(self.a(F.silu(self.n1(x))))))

class ChannelBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.n=norm(c);self.a=nn.Conv2d(c,4*c,1);self.b=nn.Conv2d(4*c,c,1)
    def forward(self,x):return x+self.b(F.silu(self.a(self.n(x))))

class SpatialBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.n=norm(c);self.dw=nn.Conv2d(c,c,5,padding=2,groups=c);self.a=nn.Conv2d(c,4*c,1);self.b=nn.Conv2d(4*c,c,1);self.scale=nn.Parameter(torch.full((1,c,1,1),.01))
    def forward(self,x):return x+self.scale*self.b(F.silu(self.a(self.dw(self.n(x)))))

class TemporalBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.n1=norm(c);self.n2=norm(c);self.pre=nn.Conv2d(c,c,1);self.local=nn.Conv2d(c,c,5,padding=2,groups=c);self.dilated=nn.Conv2d(c,c,7,padding=9,dilation=3,groups=c);self.gate=nn.Conv2d(c,2*c,1);self.out=nn.Conv2d(c,c,1)
        self.ffn=nn.Sequential(nn.Conv2d(c,4*c,1),nn.Conv2d(4*c,4*c,3,padding=1,groups=4*c),nn.GELU(),nn.Conv2d(4*c,c,1))
        self.s1=nn.Parameter(torch.full((1,c,1,1),.01));self.s2=nn.Parameter(torch.full((1,c,1,1),.01))
    def forward(self,x):
        feature=F.gelu(self.pre(self.n1(x)))
        content,gate=self.gate(self.dilated(self.local(feature))).chunk(2,1)
        x=x+self.s1*self.out(content*gate.sigmoid());return x+self.s2*self.ffn(self.n2(x))

class Faithful(nn.Module):
    def __init__(self,mode='transport',widths=(8,16,24),latent=16,controller=16,field=16,rank=8,key=8,value=8,motion=16,coarse=8,plan=8,correction=16):
        super().__init__();self.mode=mode;self.d=latent;self.groups=4;self.heads=4;self.key=key;self.value=value;self.plan_channels=plan
        a,b,c=widths;d=latent;g=self.groups
        self.stem=nn.Conv2d(21,a,1);self.down1=nn.Conv2d(a,b,3,stride=2,padding=1);self.down2=nn.Conv2d(b,c,3,stride=2,padding=1)
        self.history_blocks=nn.Sequential(*[Residual(c) for _ in range(3)],ChannelBlock(c));self.history_norm=norm(c);self.history_out=nn.Conv2d(c,4*d+5*g+48,1)
        self.controller=nn.Sequential(nn.LayerNorm(2*d+16),nn.Linear(2*d+16,controller),nn.SiLU(),nn.Linear(controller,3*d+5*g))
        self.source=nn.Sequential(nn.Linear(d,field),nn.SiLU(),nn.Linear(field,16));self.transport=nn.Sequential(nn.Linear(d,field),nn.SiLU(),nn.Linear(field,32))
        self.feedback=nn.Linear(16,d,bias=False);self.exchange_down=nn.Linear(d,rank,bias=False);self.exchange_up=nn.Linear(rank,d,bias=False)
        self.memory_norm=nn.LayerNorm(d);self.qkv=nn.Linear(d,2*key+value,bias=False);self.memory_out=nn.Linear(value,d,bias=False);self.age_bias=nn.Parameter(torch.zeros(4,7))
        self.motion_down=nn.Linear(d,motion,bias=False);self.fine=nn.Sequential(*[SpatialBlock(motion) for _ in range(2)])
        self.coarse_down=nn.Conv2d(motion,coarse,1,bias=False);self.coarse=nn.Sequential(*[SpatialBlock(coarse) for _ in range(2)]);self.coarse_up=nn.Conv2d(coarse,motion,1,bias=False);self.motion_up=nn.Linear(motion,d,bias=False)
        self.plan_in=nn.Conv2d(d,plan,1);self.planner=nn.Sequential(*[TemporalBlock(plan) for _ in range(4)]);self.plan_norm=norm(plan);self.plan_out=nn.Linear(plan,10*plan);self.aux=nn.Linear(plan,1)
        self.correction_features=nn.Sequential(nn.LayerNorm(d+plan),nn.Linear(d+plan,correction),nn.SiLU());self.correction_source=nn.Linear(correction,16);self.correction_transport=nn.Linear(correction,32)
        # Same initialization rules as reference; independent parameter naming/RNG order.
        nn.init.normal_(self.history_out.weight,std=.005);nn.init.zeros_(self.history_out.bias)
        with torch.no_grad():
            self.history_out.weight[:d].normal_(std=.02);self.history_out.bias[d:d+d//2].copy_(torch.linspace(.03,.5,d//2));self.history_out.bias[d+d//2:2*d].fill_(-4);self.history_out.bias[4*d:4*d+5*g:5].fill_(3);self.history_out.weight[-48:].normal_(std=.001)
        for head in [self.source,self.transport]:nn.init.normal_(head[-1].weight,std=.002);nn.init.zeros_(head[-1].bias)
        nn.init.normal_(self.controller[-1].weight,std=.001);nn.init.zeros_(self.controller[-1].bias)
        for layer,std in [(self.feedback,.01),(self.exchange_down,.02),(self.exchange_up,.002),(self.aux,.02)]:nn.init.normal_(layer.weight,std=std)
        nn.init.zeros_(self.aux.bias)
        for layer in [self.correction_source,self.correction_transport]:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    @staticmethod
    def pack(image):return F.pixel_unshuffle(image,4).permute(0,2,3,1).contiguous()
    def prepare(self,context):
        b,t,_,h,w=context.shape;images=context[:,:,0]
        x=(torch.arange(w,device=context.device)+.5)*2/w-1;y=(torch.arange(h,device=context.device)+.5)*2/h-1
        coords=torch.stack(torch.meshgrid(y,x,indexing='ij')[::-1])[None].expand(b,-1,-1,-1)
        x=torch.cat((images,images[:,1:]-images[:,:-1],coords),1)
        x=F.leaky_relu(self.stem(x),.2);x=F.leaky_relu(self.down1(x),.2);x=F.leaky_relu(self.down2(x),.2)
        encoded=self.history_out(F.silu(self.history_norm(self.history_blocks(x)))).permute(0,2,3,1)
        d=self.d;z=encoded[...,:d];controls=encoded[...,d:4*d];logits=encoded[...,4*d:4*d+20].reshape(b,h//4,w//4,4,5);offset=encoded[...,-48:]
        planned=self.plan_norm(self.planner(self.plan_in(z.permute(0,3,1,2))))
        planned=self.plan_out(planned.permute(0,2,3,1)).reshape(b,h//4,w//4,10,self.plan_channels).permute(0,3,1,2,4).contiguous()
        return z,controls,logits,offset,self.pack(context[:,-1]),planned
    def encode(self,z):return self.qkv(self.memory_norm(z)).split((self.key,self.key,self.value),-1)
    def recall(self,z,q,keys,values):
        q=q.unflatten(-1,(4,self.key//4));keys=keys.unflatten(-1,(4,self.key//4)).transpose(-3,-2);values=values.unflatten(-1,(4,self.value//4)).transpose(-3,-2)
        score=(q.unsqueeze(-2)*keys).sum(-1)/math.sqrt(self.key//4)
        recalled=((score+self.age_bias).softmax(-1).unsqueeze(-1)*values).sum(-2).flatten(-2)
        z=z+.1*self.memory_out(recalled);x=self.motion_down(z).permute(0,3,1,2)
        low=self.coarse_down(F.avg_pool2d(x,2));delta=self.coarse_up(self.coarse(low)-low)
        delta=F.interpolate(delta,size=x.shape[-2:],mode='bilinear',align_corners=False)+self.fine(x)-x
        return z+self.motion_up(delta.permute(0,2,3,1))
    def controls(self,z,image,z0,base,logits,anchor):
        correction=self.controller(torch.cat((z,z0,self.pack(image)-anchor),-1));d=self.d
        omega,decay,feedback,drift=(base+correction[...,:3*d]).split((d//2,d//2,d,d),-1)
        rho=torch.exp(-.5*F.softplus(decay));rotation=torch.stack((rho*torch.cos(.5*omega),rho*torch.sin(.5*omega)),-1)
        weights=(logits+correction[...,3*d:].reshape(*z.shape[:-1],4,5)).softmax(-1)
        return rotation,weights,feedback.sigmoid(),.5*drift
    def move(self,z,image,controls,anchor):
        rotation,weights,feedback,drift=controls;z=z+.1*self.exchange_up(self.exchange_down(z))
        grouped=z.reshape(*z.shape[:-1],4,self.d//4)
        neighbors=[grouped,torch.cat((grouped[:,:,:1],grouped[:,:,:-1]),2),torch.cat((grouped[:,:,1:],grouped[:,:,-1:]),2),torch.cat((grouped[:,:1],grouped[:,:-1]),1),torch.cat((grouped[:,1:],grouped[:,-1:]),1)]
        mixed=sum(weights[...,i:i+1]*v for i,v in enumerate(neighbors)).reshape_as(z)
        pairs=mixed.reshape(*mixed.shape[:-1],self.d//2,2);c,s=rotation[...,0],rotation[...,1];a,b=pairs[...,0],pairs[...,1]
        return torch.stack((c*a-s*b,s*a+c*b),-1).flatten(-2)+feedback*self.feedback(self.pack(image)-anchor)+drift
    def forward(self,context):
        z0,base,logits,offset,anchor,plan=self.prepare(context);z=z0;image=context[:,-1]
        q,k,v=self.encode(z);keys=torch.stack([k]*7,-2);values=torch.stack([v]*7,-2)
        coarse=self.aux(plan).permute(0,1,4,2,3);pred=[];energies=[];variations=[]
        for frame in range(10):
            z=self.recall(z,q,keys,values);control=self.controls(z,image,z0,base,logits,anchor)
            for half in range(2):
                correction=self.correction_features(torch.cat((z,plan[:,frame]),-1))
                packed=torch.cat((self.source(z),self.transport(z)),-1)+offset+torch.cat((self.correction_source(correction),self.correction_transport(correction)),-1)
                fields=F.pixel_shuffle(packed.permute(0,3,1,2).contiguous(),4);r=fields[:,:1];w=fields[:,1:]
                if self.mode=='transport_off':w=w*0
                energies.append(r.square().mean());variations.append(total_variation(w))
                image=exponential(image,r,w) if self.mode=='transport' else image+.5*r
                if frame<9 or half<1:z=self.move(z,image,control,anchor)
            pred.append(image)
            if frame<9:
                q,k,v=self.encode(z);keys=torch.cat((keys[...,:1,:],keys[...,2:,:],k.unsqueeze(-2)),-2);values=torch.cat((values[...,:1,:],values[...,2:,:],v.unsqueeze(-2)),-2)
        return torch.stack(pred,1),torch.stack(energies).mean(),torch.stack(variations).mean(),coarse
