from typing import Callable
import torch


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, activation:Callable):
        return activation(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class FullLikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fill_value:float=1.0):
        # Save input shape for backward pass
        ctx.save_for_backward(x)
        # Return tensor filled with fill_value in same shape as input
        return torch.full_like(x, fill_value)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient through unchanged
        return grad_output, None
        

def full_like_ste(x:torch.Tensor, fill_value:float=1.0):
    return FullLikeSTE.apply(x, fill_value)


def ones_like_ste(x:torch.Tensor):
    return FullLikeSTE.apply(x, 1.0)


def zeros_like_ste(x:torch.Tensor):
    return FullLikeSTE.apply(x, 0.0)


def sigmoid_ste(x:torch.Tensor):
    return STE.apply(x, torch.sigmoid)

