import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.autograd.function import InplaceFunction, Function
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time
import random
import BlockQuantizeCUDA as BQC

class _qblock_linear(Function):
    @staticmethod
    def forward(ctx, X8_3D, SX16, W32, bias, B):
        '''
            X8_3D: (BS, N, C_in),
            SX_16: (BS, N / B, C_in / B),
            W32: (C_out, C_in),
            bias: (C_out),
            B: Blocksize = 32

            O8_3D: (BS, N, C_out), row major
            SO16: (BS, C_out / B, N / B), row major
        '''
        X_3D_shape = X_3D.shape
        O_3D_shape = list(X_3D_shape)
        O_3D_shape[-1] = W32.shape[0]

        C_in = X_3D.shape[-1]
        
        X8 = X_3D.reshape(-1, C_in)
        W8, SW16 = int8_quantize(W32, Blocksize) # W8 is (C_out, C_in), row major, SW16 should be (C_out / B, C_in / B)
        SX16T = SX16.t().contiguous()
        SW16T = SW16.t().contiguous()
    
        biasmax = bias.max()
        O8, SO16 = BQC.igemm_output_int_quantize_bias_rowcol(X8, W8.t(), bias, biasmax, SX16T, SW16T, X8.shape[0], W8.shape[0], X8.shape[1])

        O8_3D = O8.reshape(O_3D_shape)
        ctx.saved = X8, SX16, W8, SW16

        return O8_3D, SO16

    @staticmethod
    def backward(ctx, G8_3D, SG16):
        '''
            G8_3D: (BS, N, C_out)
            SG16: (BS, N / B, C_out / B)
        '''
        X8, SX16, W8, SW16 = ctx.saved
        SG16T = SG16.t().contiguous()
        
        C_out = grad_output.shape[-1]
        G8 = G8_3D.reshape(-1, C_out)

        G8T = transpose(G8)

        GW = BQC.igemm_output_fp_no_quantize(G8T, X8, SG16, SX16, G8T.shape[0], X8.shape[1], G8T.shape[1])
        GA8, GA16 = BQC.igemm_output_int_quantize_stochastic(G8, W8, SG16T, SW16, G8.shape[0], W8.shape[1], G8.shape[1])

        if bias is not None:
            grad_bias = G8.sum(0)
        else:
            grad_bias = None

        grad_input = grad_input_flatten.reshape(input_shape)

        return  GW, grad_bias, None

class QBlockLinear(nn.Linear):
    """Block Quantize Linear Layer"""

    def __init__(self, in_features, out_features, bias=True, args=None, layer_type=''):
        super(QBlockLinear, self).__init__(in_features, out_features, bias)
        
        self.args = args
        self.layer_type = layer_type
    
    def forward(self, X83D, SX16)
        output = _qblock_linear.apply(X8_3D, SX16, self.weight, self.bias, self.args.B)
        return output
