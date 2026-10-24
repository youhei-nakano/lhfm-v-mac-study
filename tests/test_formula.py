import unittest,math
import torch
from operators import generator,exponential,objective,schedule
from architecture import Faithful

class FormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_dense_stencil_and_affine_matrix_exponential(self):
        torch.manual_seed(1);n=4;w=torch.randn(1,2,n,n,dtype=torch.float64)*.7
        # Independent scalar formula, zero exterior, x then y.
        def scalar(a):
            out=torch.zeros_like(a)
            for y in range(n):
                for x in range(n):
                    for c,(dx,dy) in enumerate([(1,0),(0,1)]):
                        speed=w[0,c,y,x];sign=1 if speed>=0 else -1
                        sample=lambda xx,yy:a[yy,xx] if 0<=xx<n and 0<=yy<n else 0.
                        out[y,x]+=abs(speed)*(-1.5*a[y,x]+2*sample(x-sign*dx,y-sign*dy)-.5*sample(x-2*sign*dx,y-2*sign*dy))
            return out
        eye=torch.eye(n*n,dtype=torch.float64);a=torch.stack([scalar(row.reshape(n,n)).flatten() for row in eye],1)
        image=torch.randn(1,1,n,n,dtype=torch.float64);r=torch.randn_like(image)*.2
        torch.testing.assert_close(generator(image,w)[0,0],scalar(image[0,0]),atol=1e-14,rtol=1e-14)
        augmented=torch.zeros(n*n+1,n*n+1,dtype=torch.float64);augmented[:-1,:-1]=a;augmented[:-1,-1]=r.flatten()
        initial=torch.cat((image.flatten(),torch.ones(1,dtype=torch.float64)))
        exact=(torch.matrix_exp(.5*augmented)@initial)[:-1].reshape_as(image)
        torch.testing.assert_close(exponential(image,r,w),exact,atol=2e-10,rtol=2e-10)
    def test_zero_limits_and_adaptive_pieces(self):
        x=torch.randn(1,1,4,4,dtype=torch.float64);r=torch.randn_like(x);w=torch.zeros(1,2,4,4,dtype=torch.float64)
        torch.testing.assert_close(exponential(x,r,w),x+.5*r,atol=1e-13,rtol=1e-13)
        # dt splitting with frozen fields agrees independently at high field rate.
        w.fill_(7.1);a=exponential(x,r,w,.5);b=exponential(exponential(x,r,w,.25),r,w,.25)
        torch.testing.assert_close(a,b,atol=2e-9,rtol=2e-9)
    def test_gradient_all_physical_inputs(self):
        x=torch.randn(1,1,3,3,dtype=torch.float64,requires_grad=True);r=torch.randn_like(x,requires_grad=True)
        w=(torch.rand(1,2,3,3,dtype=torch.float64)+.2).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda a,b,c:exponential(a,b,c),(x,r,w),eps=1e-6,atol=2e-5,rtol=2e-4))
    def test_full_architecture_semantics_and_losses(self):
        torch.manual_seed(2);m=Faithful();x=torch.rand(1,10,1,16,16);target=torch.rand(1,10,1,16,16)
        p,e,t,c=m(x);self.assertEqual(p.shape,target.shape);self.assertEqual(c.shape,(1,10,1,4,4))
        loss,diag=objective(p,target,e,t,c);torch.testing.assert_close(loss,diag['mse']+.001*e+.0001*t+.05*diag['coarse_mse'])
        loss.backward();self.assertTrue(all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters()))
        self.assertGreater(m.aux.weight.grad.abs().sum(),0)
        self.assertEqual(m.planner.__len__(),4);self.assertEqual(m.age_bias.shape,(4,7))
    def test_schedule_endpoints_and_anchor(self):
        self.assertAlmostEqual(schedule(1,1250000)[0],4e-5,places=12)
        self.assertAlmostEqual(schedule(375000,1250000)[0],.001,places=12)
        self.assertAlmostEqual(schedule(1250000,1250000)[0],4e-9,places=14)
        self.assertLess(schedule(400001,1250000)[0],schedule(400000,1250000)[0]*.501)
        self.assertAlmostEqual(schedule(40,40)[0],4e-9,places=14)


class FidelityTests(unittest.TestCase):
    def test_original_width_parameter_count(self):
        m=Faithful(widths=(64,128,448),latent=384,controller=256,field=192,rank=128,key=128,value=256,motion=320,coarse=192,plan=128,correction=192)
        self.assertEqual(sum(v.numel() for v in m.parameters()),18637493)
    def test_cadence_and_memory_components(self):
        m=Faithful();counts={};handles=[]
        for name in ['planner','controller','source','feedback','qkv','aux']:
            counts[name]=0
            def hook(layer,args,out,n=name):counts[n]+=1
            handles.append(getattr(m,name).register_forward_hook(hook))
        with torch.no_grad():m(torch.rand(1,10,1,16,16))
        for h in handles:h.remove()
        self.assertEqual(counts,{'planner':1,'controller':10,'source':20,'feedback':19,'qkv':10,'aux':1})
    def test_paper_ssim_identity_and_error_normalization(self):
        import numpy as np
        from metrics import evaluate
        rng=np.random.default_rng(3);a=rng.random((1,10,1,64,64),dtype=np.float32)
        result=evaluate(a,a)
        self.assertEqual(result['mse_raw'],0);self.assertEqual(result['ssim_range2'],1);self.assertEqual(result['psnr_clipped'],120)
        result=evaluate(a+.1,a)
        self.assertAlmostEqual(result['mse_spatial_sum_raw'],result['mse_raw']*4096,places=12)

if __name__=='__main__':unittest.main()
