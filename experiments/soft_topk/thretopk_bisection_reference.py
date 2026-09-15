"""Correctness reference only: per-row normalization + FP64 60-step bisection.

Not used for any historical ImageNet result. Not a performance implementation.
The implicit backward differentiates the exact soft-cardinality constraint.
"""
import math
import torch


class _SoftCardinalityBisection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        xd = x.to(torch.float64)
        n = xd.shape[-1]
        margin = math.log(2.0 * n)
        lo = xd.amin(-1, keepdim=True) - margin
        hi = xd.amax(-1, keepdim=True) + margin
        for _ in range(60):
            mid = (lo + hi) * 0.5
            delta = xd - mid
            tail = 0.5 * torch.exp(-delta.abs())
            y = torch.where(delta > 0, 1.0 - tail, tail)
            too_many = y.sum(-1, keepdim=True) > k
            lo = torch.where(too_many, mid, lo)
            hi = torch.where(too_many, hi, mid)
        delta = xd - (lo + hi) * 0.5
        hprime = 0.5 * torch.exp(-delta.abs())
        y = torch.where(delta > 0, 1.0 - hprime, hprime)
        ctx.save_for_backward(hprime)
        ctx.input_dtype = x.dtype
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (hprime,) = ctx.saved_tensors
        gd = grad_output.to(torch.float64)
        weighted_mean = (gd * hprime).sum(-1, keepdim=True) / hprime.sum(-1, keepdim=True)
        grad_x = hprime * (gd - weighted_mean)
        return grad_x.to(ctx.input_dtype), None


def thretopk_bisection_reference(x, k, temperature=30.0, eps=None):
    """Return [B,N] soft weights with per-row sum k (within output precision).

    The forward solves in FP64 for 60 iterations. Gradients through rowwise
    normalization are handled by PyTorch; threshold gradients use implicit VJP.
    Finite floating inputs and 0 <= integer k <= N are the supported domain.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() != 2 or not x.is_floating_point():
        raise ValueError('Expected floating [B,N] or [N] input')
    n = x.shape[-1]
    if not isinstance(k, int) or not 0 <= k <= n:
        raise ValueError('k must be an integer from 0 to N')
    if k == 0:
        return x * 0.0
    if k == n:
        return x * 0.0 + 1.0
    floor = torch.finfo(x.dtype).eps if eps is None else eps
    if floor <= 0:
        raise ValueError('eps must be positive')
    row_min = x.amin(-1, keepdim=True)
    row_range = x.amax(-1, keepdim=True) - row_min
    normalized = (x - row_min) / row_range.clamp_min(floor) * temperature
    return _SoftCardinalityBisection.apply(normalized, k)
