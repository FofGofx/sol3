"""LTX Stage 1 video-block pointwise kernels and split RoPE."""

import torch
import triton
import triton.language as tl


from ._pointwise import _gate_index32 as _gate, _mod_index32 as _mod, _rope


FMA = True

@triton.jit
def split_rope(X,C,S,O,B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,X0:tl.constexpr,XT:tl.constexpr,XH:tl.constexpr,XD:tl.constexpr,C0:tl.constexpr,CH:tl.constexpr,CT:tl.constexpr,CD:tl.constexpr,S0:tl.constexpr,SH:tl.constexpr,ST:tl.constexpr,SD:tl.constexpr,THREE:tl.constexpr,FUSED:tl.constexpr,BLOCK:tl.constexpr=256):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);live=i<B*T*H*D
 d=i%D
 if THREE:h=i//D%H;t=i//(D*H)%T;b=i//(D*H*T)
 else:t=i//D%T;h=i//(D*T)%H;b=i//(D*T*H)
 other=tl.where(d<D//2,d+D//2,d-D//2)
 x=tl.load(X+b.to(tl.int64)*X0+t.to(tl.int64)*XT+h*XH+d*XD,live,0).to(tl.float32)
 y=tl.load(X+b.to(tl.int64)*X0+t.to(tl.int64)*XT+h*XH+other*XD,live,0).to(tl.float32)
 c=tl.load(C+b.to(tl.int64)*C0+h*CH+t.to(tl.int64)*CT+(d%(D//2))*CD,live,0).to(tl.float32)
 s=tl.load(S+b.to(tl.int64)*S0+h*SH+t.to(tl.int64)*ST+(d%(D//2))*SD,live,0).to(tl.float32)
 s=tl.where(d<D//2,-s,s)
 first=(x*c).to(X.dtype.element_ty).to(tl.float32)
 if FUSED:out=tl.fma(s,y,first)
 else:out=first+s*y
 tl.store(O+i,out,live)


def fused(x,c,s):
 three=x.ndim==3
 if three:b,t,hd=x.shape;h=c.shape[1];d=hd//h;xs=(x.stride(0),x.stride(1),d*x.stride(2),x.stride(2))
 else:b,h,t,d=x.shape;xs=(x.stride(0),x.stride(2),x.stride(1),x.stride(3))
 cs=(0 if c.shape[0]==1 else c.stride(0),c.stride(1),c.stride(2),c.stride(3))
 ss=(0 if s.shape[0]==1 else s.stride(0),s.stride(1),s.stride(2),s.stride(3))
 out=torch.empty(x.shape,device=x.device,dtype=x.dtype)
 split_rope[(triton.cdiv(x.numel(),256),)](x,c,s,out,b,t,h,d,*xs,*cs,*ss,three,FMA,num_warps=4,enable_fp_fusion=False)
 return out

KERNELS = {'pointwise': ('_mod', '_gate', '_rope'), 'rotary': ('split_rope',)}
