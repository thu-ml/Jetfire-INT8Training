import torch
import BlockQuantizeCUDA as BQC

def int8_linear_forward(X8_3D, SX16_3D, W8, SW16, output_quantized, B):
    X8_3D_shape = X8_3D.shape
    SX16_3D_shape = SX16_3D.shape
    O_3D_shape = list(X8_3D_shape)
    SO16_3D_shape = list(SX16_3D_shape)
    O_3D_shape[-1] = W8.shape[0]

    C_in = X8_3D.shape[-1]
    
    X8 = X8_3D.reshape(-1, C_in)
    SO16_3D_shape[-1] = SW16.shape[0]

    SX16 = SX16_3D.reshape(-1, SX16_3D.shape[-1])
    SX16T = SX16.t().contiguous()
    SW16T = SW16.t().contiguous()

    bias = torch.zeros((W8.shape[0]), device=W8.device)
    biasmax = torch.zeros((1,), device=W8.device)

    O8, SO16 = BQC.igemm_output_int_quantize_bias_rowcol(X8, W8.t(), bias, biasmax, SX16T, SW16T, X8.shape[0], W8.shape[0], X8.shape[1])

    if output_quantized:
        O8_3D = O8.reshape(O_3D_shape)
        SO16_3D = SO16.reshape(SO16_3D_shape)
        return O8_3D, SO16_3D
    else:
        O16 = O8.reshape(O8.shape[0] // B, B, O8.shape[1] // B, B) * SO16.reshape(O8.shape[0] // B, O8.shape[1] // B).unsqueeze(1).unsqueeze(3)
        O16 = O16.reshape(O8.shape).reshape(O_3D_shape)
        return O16

