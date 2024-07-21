from typing import Tuple
import torch
from torch import autograd
import torch.nn as nn
import triton
import triton.language as tl

import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from fgelu import int8_gelu_forward
from bgelu import int8_gelu_backward
from dequantize import int8_dequantize
from quantize_s import int8_quantize_stochastic
from quantize_d import int8_quantize_deterministic

import numpy as np


def torch_gelu_backward(x): 
    cdf = 0.5 * (1.0 + torch.erf(x / np.sqrt(2)))
    exp = x / (np.sqrt(2 * np.pi)) * torch.exp(-x ** 2 / 2)
    return cdf + exp
    
class _GeLU(autograd.Function):
    """
    Autograd wrapper for triton implementation.
    """

    @staticmethod
    def forward(
            ctx, x, B
    ) -> torch.Tensor:
        """
        Forward pass.
        :param ctx: Context variable
        :param qx (torch.Tensor): Input Int8 tensor of any shape
        :param sx (torch.Tensor): Fp16 scalar tensor
        :return (torch.Tensor): Activation tensor of the same shape as the input
        """
        # Save input tensor and beta value for backward pass
        qx, sx = int8_quantize_deterministic(x, B)
        ctx.saved = sx, qx, B

        # Compute output activation
        y, s_y = int8_gelu_forward(qx, sx, B)
        out = int8_dequantize(y, s_y, B)
        return out

    @staticmethod
    def backward(
            ctx, grad_out
            
    ) -> Tuple[torch.Tensor, None]:
        """
        Backward pass.
        :param ctx: Context variable
        :param grad_output (torch.Tensor): Previous gradient
        :return (Tuple[torch.Tensor, None]): Gradient of input
        """
        # Get saved variables
        sx, qx, B = ctx.saved

        grad_y, grad_sy = int8_quantize_stochastic(grad_out, B)
        
        # Compute gradient with triton kernel
        grad_qx, grad_sx = int8_gelu_backward(qx, sx, grad_y, grad_sy, B) # sx.shape, grad_sx.shape torch.Size([5, 64, 64]) torch.Size([5, 2, 2])

        grad_x = int8_dequantize(grad_qx, grad_sx, B)

        import IPython
        IPython.embed()

        return grad_x, None #下一层往上传的梯度


# Make autograd function
gelu_function = _GeLU.apply


class QBlockGeLU(nn.Module):

    def __init__(self):
        # Call super constructor
        super(QBlockGeLU, self).__init__()

    def forward(self, x, B) -> torch.Tensor:
        """
        Forward pass.
        :param input (torch.Tensor): Tensor of any shape
        :return (torch.Tensor): Output activation tensor of the same shape as the input tensor
        """
        return gelu_function(x, B)
    

if __name__ == "__main__":
    # x = torch.tensor([1., 2., 3.], requires_grad=True)
    
    torch.set_printoptions(edgeitems=2, precision=5, sci_mode=False)
    torch.manual_seed(0)

    BS=1
    SL=64
    CDIM=64
    QB=32
    
    qx = (torch.randn((BS, SL, CDIM)).cuda() * 254 - 127).to(torch.int8)

    # triton result
    _qx = qx.reshape(BS, SL // QB, QB, CDIM // QB, QB).permute(0, 1, 3, 2, 4)
    sx = torch.randn(BS, SL // QB, CDIM // QB).cuda().to(torch.float16)
    qx = _qx.permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)
    _rqx = (_qx * sx.unsqueeze(3).unsqueeze(4))
    rqx = _rqx.permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM).to(torch.float32) / 50
    rqx.requires_grad_()
    rqx_self = rqx.clone().detach().requires_grad_(True)

    torch_gelu = nn.GELU()
    output_torch = torch_gelu(rqx)
    loss_torch = output_torch.mean() * 1e7
    loss_torch.backward()

    self_gelu = QBlockGeLU()
    output_triton = self_gelu(rqx_self, QB)
    loss_triton = output_triton.mean() * 1e7
    loss_triton.backward()

    import IPython
    IPython.embed()