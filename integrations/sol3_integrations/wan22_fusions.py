"""Wan high- and low-noise expert input and residual fusions."""

import math
from torch import nn
import torch
import triton
import triton.language as tl


@triton.jit
def _mod(X,S,B,Y,T:tl.constexpr,D:tl.constexpr,N:tl.constexpr,PER_TOKEN:tl.constexpr,BLOCK:tl.constexpr):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);r=ix//D;d=ix%D
 param=r*D+d if PER_TOKEN else (r//T)*D+d
 x=tl.load(X+ix,ix<N,0).to(tl.float32);s=tl.load(S+param,ix<N,0).to(tl.float32);b=tl.load(B+param,ix<N,0).to(tl.float32)
 out=(x*(1.+s))+b;tl.store(Y+ix,out,ix<N)


@triton.jit
def _gate(X,A,G,Y,T:tl.constexpr,D:tl.constexpr,N:tl.constexpr,PER_TOKEN:tl.constexpr,BLOCK:tl.constexpr):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);r=ix//D;d=ix%D;param=r*D+d if PER_TOKEN else (r//T)*D+d
 x=tl.load(X+ix,ix<N,0).to(tl.float32);a=tl.load(A+ix,ix<N,0).to(tl.float32);g=tl.load(G+param,ix<N,0).to(tl.float32)
 out=x+(a*g);tl.store(Y+ix,out,ix<N)


@triton.jit
def _rope(Q,K,C,S,QO,KO,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,N:tl.constexpr,FREQ_BF16:tl.constexpr,BLOCK:tl.constexpr):
 pair=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);feature=(pair*2)%D;token=(pair*2//(H*D))%T
 ix=pair*2;mask=ix<N;freq=token*D+feature
 c=tl.load(C+freq,mask,0);s=tl.load(S+freq+1,mask,0)
 q1=tl.load(Q+ix,mask,0).to(c.dtype);q2=tl.load(Q+ix+1,mask,0).to(c.dtype);k1=tl.load(K+ix,mask,0).to(c.dtype);k2=tl.load(K+ix+1,mask,0).to(c.dtype)
 if FREQ_BF16:
  qc=(q1.to(tl.float32)*c.to(tl.float32)).to(tl.bfloat16).to(tl.float32);qs=(q2.to(tl.float32)*s.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
  qr=(q1.to(tl.float32)*s.to(tl.float32)).to(tl.bfloat16).to(tl.float32);qi=(q2.to(tl.float32)*c.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
  kc=(k1.to(tl.float32)*c.to(tl.float32)).to(tl.bfloat16).to(tl.float32);ks=(k2.to(tl.float32)*s.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
  kr=(k1.to(tl.float32)*s.to(tl.float32)).to(tl.bfloat16).to(tl.float32);ki=(k2.to(tl.float32)*c.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
 else:
  qc=q1*c;qs=q2*s;qr=q1*s;qi=q2*c;kc=k1*c;ks=k2*s;kr=k1*s;ki=k2*c
 tl.store(QO+ix,(qc-qs).to(tl.float32),mask);tl.store(QO+ix+1,(qr+qi).to(tl.float32),mask);tl.store(KO+ix,(kc-ks).to(tl.float32),mask);tl.store(KO+ix+1,(kr+ki).to(tl.float32),mask)


def mod(x,scale,shift,dtype):
 x=x.contiguous();b,t,d=x.shape;y=torch.empty_like(x,dtype=dtype);per=scale.shape[1]!=1
 _mod[(triton.cdiv(x.numel(),1024),)](x,scale.contiguous(),shift.contiguous(),y,t,d,x.numel(),per,1024,enable_fp_fusion=False);return y


def gate(x,a,g):
 x=x.contiguous();a=a.contiguous();b,t,d=x.shape;y=torch.empty_like(x)
 _gate[(triton.cdiv(x.numel(),1024),)](x,a,g.contiguous(),y,t,d,x.numel(),g.shape[1]!=1,1024,enable_fp_fusion=False);return y


def rope(q,k,cos,sin):
 assert q.is_contiguous() and k.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
 b,t,h,d=q.shape;qo=torch.empty_like(q);ko=torch.empty_like(k)
 _rope[(triton.cdiv(q.numel()//2,1024),)](q,k,cos,sin,qo,ko,t,h,d,q.numel(),cos.dtype==torch.bfloat16,1024,enable_fp_fusion=False);return qo,ko

LAST = {}
_FAILED = set()
_EPS = {}

def eligibility(q,k,norm_q,norm_k,cos,sin,heads,*,require_cuda=True):
    if torch.is_autocast_enabled('cuda'):return False,'cuda_autocast_enabled'
    if torch.__version__.split('+')[0]!='2.5.1':return False,'unsupported_torch_rmsnorm_version'
    if q.ndim!=3 or k.shape!=q.shape:return False,'projection_shape'
    if require_cuda and q.device.type!='cuda':return False,'non_CUDA'
    if q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or k.device!=q.device:return False,'projection_dtype_or_device'
    if torch.is_grad_enabled():return False,'autograd_enabled'
    if q.stride(-1)!=1 or k.stride(-1)!=1:return False,'feature_stride'
    b,t,c=q.shape
    if b<1 or t<1 or c<1:return False,'empty_projection'
    if type(heads) is not int or heads<1 or c%heads or c//heads!=128 or c>8192:return False,'head_layout'
    for norm in (norm_q,norm_k):
        if type(norm) is not nn.RMSNorm or 'forward' in norm.__dict__ or norm._forward_hooks or norm._forward_pre_hooks:return False,'nonstandard_RMSNorm'
        if tuple(norm.normalized_shape)!=(c,) or norm.weight is None:return False,'normalization_scope'
        w=norm.weight
        if w.shape!=(c,) or w.dtype!=torch.bfloat16 or w.device!=q.device or not w.is_contiguous():return False,'normalization_weight'
        if norm.eps is not None and (type(norm.eps) not in (int,float) or not math.isfinite(norm.eps) or norm.eps<=0):return False,'epsilon'
    if not isinstance(cos,torch.Tensor) or not isinstance(sin,torch.Tensor):return False,'rotary_missing'
    if cos.shape!=(1,t,1,128) or sin.shape!=cos.shape:return False,'rotary_shape'
    if cos.device!=q.device or sin.device!=q.device or sin.dtype!=cos.dtype or cos.dtype not in (torch.bfloat16,torch.float32,torch.float64):return False,'rotary_dtype_or_device'
    if not cos.is_contiguous() or not sin.is_contiguous():return False,'rotary_stride'
    return True,'supported_metadata'

@triton.jit
def _rms_rope_exact(Q,K,IQ,IK,WQ,WK,COS,SIN,QO,KO,
                    T:tl.constexpr,C:tl.constexpr,DH:tl.constexpr,
                    QSB:tl.constexpr,QST:tl.constexpr,KSB:tl.constexpr,KST:tl.constexpr,
                    FREQ_BF16:tl.constexpr,BC:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);batch=row//T;token=row%T
    c=tl.arange(0,BC);valid=c<C
    q=tl.load(Q+batch*QSB+token*QST+c,valid,0).to(tl.float32)
    k=tl.load(K+batch*KSB+token*KST+c,valid,0).to(tl.float32)
    iq=tl.load(IQ+row).to(tl.float32);ik=tl.load(IK+row).to(tl.float32)
    wq=tl.load(WQ+c,valid,0).to(tl.float32);wk=tl.load(WK+c,valid,0).to(tl.float32)
    qn=((q*iq).to(tl.bfloat16).to(tl.float32)*wq).to(tl.bfloat16)
    kn=((k*ik).to(tl.bfloat16).to(tl.float32)*wk).to(tl.bfloat16)
    q0,q1=tl.split(tl.reshape(qn,(BC//2,2)));k0,k1=tl.split(tl.reshape(kn,(BC//2,2)))
    pairs=tl.arange(0,BC//2);even=pairs*2;live=even<C
    freq=token*DH+(even%DH)
    cs=tl.load(COS+freq,live,0);sn=tl.load(SIN+freq+1,live,0)
    q0=q0.to(cs.dtype);q1=q1.to(cs.dtype);k0=k0.to(cs.dtype);k1=k1.to(cs.dtype)
    if FREQ_BF16:
        qc=(q0.to(tl.float32)*cs.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        qs=(q1.to(tl.float32)*sn.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        qr=(q0.to(tl.float32)*sn.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        qi=(q1.to(tl.float32)*cs.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        kc=(k0.to(tl.float32)*cs.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        ks=(k1.to(tl.float32)*sn.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        kr=(k0.to(tl.float32)*sn.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        ki=(k1.to(tl.float32)*cs.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    else:
        qc=q0*cs;qs=q1*sn;qr=q0*sn;qi=q1*cs
        kc=k0*cs;ks=k1*sn;kr=k0*sn;ki=k1*cs
    tl.store(QO+row*C+even,(qc-qs).to(tl.float32),live)
    tl.store(QO+row*C+even+1,(qr+qi).to(tl.float32),live)
    tl.store(KO+row*C+even,(kc-ks).to(tl.float32),live)
    tl.store(KO+row*C+even+1,(kr+ki).to(tl.float32),live)


def _epsilon(norm):
    value=torch.finfo(torch.bfloat16).eps if norm.eps is None else float(norm.eps)
    if value not in _EPS:_EPS[value]=float(torch.tensor(value,dtype=torch.bfloat16,device='cpu'))
    return _EPS[value]


def native_inverse(x,norm):
    """ATen2.5.1 rms_norm's BF16 pow/mean/add_/rsqrt, unchanged operators."""
    return torch.rsqrt(x.pow(2).mean(-1,keepdim=True).add_(_epsilon(norm)))


def maybe_fused(q,k,norm_q,norm_k,cos,sin,heads,*,warps=8,strict=False):
    valid,reason=eligibility(q,k,norm_q,norm_k,cos,sin,heads)
    LAST.clear();LAST.update(used=False,reason=reason)
    if not valid:return None
    b,t,c=q.shape;signature=(q.device,tuple(q.shape),q.stride(),k.stride(),cos.dtype,norm_q.eps,norm_k.eps,warps)
    if signature in _FAILED and not strict:
        LAST.update(reason='previous_compile_or_resource_failure');return None
    try:
        iq=native_inverse(q,norm_q);ik=native_inverse(k,norm_k)
        assert iq.dtype==ik.dtype==torch.bfloat16 and iq.is_contiguous() and ik.is_contiguous()
        qo=torch.empty((b,t,heads,128),device=q.device,dtype=q.dtype);ko=torch.empty_like(qo)
        _rms_rope_exact[(b*t,)](q,k,iq,ik,norm_q.weight,norm_k.weight,cos,sin,qo,ko,t,c,128,q.stride(0),q.stride(1),k.stride(0),k.stride(1),cos.dtype==torch.bfloat16,triton.next_power_of_2(c),num_warps=warps,enable_fp_fusion=False)
        LAST.update(used=True,reason='native_bf16_rmsnorm',shape=list(q.shape),freq_dtype=str(cos.dtype),warps=warps,native_BF16_pow_mean_add_rsqrt=True,input_multiply_weight_and_rotary_BF16_rounds=True,all_address_arithmetic_64bit=True)
        return qo,ko
    except Exception as error:
        _FAILED.add(signature);LAST.update(reason='compile_or_resource_failure',error=type(error).__name__+': '+str(error))
        if strict:raise
        return None

KERNELS = {'pointwise': ('_mod', '_gate', '_rope'), 'rms_rope': ('_rms_rope_exact',)}
