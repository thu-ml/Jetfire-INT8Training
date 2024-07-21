import torch
import BlockQuantizeCUDA as BQC

def int8_linear_backward(X8_3D, SX16_3D, G8_3D, SG16_3D, G8T, W8, SW16, B, stochastic=True, dgrad_quantize=True):
    GX8_3D_shape = list(G8_3D.shape)
    GX8_3D_shape[-1] = W8.shape[1]
    GSX16_3D_shape = list(SG16_3D.shape)
    GSX16_3D_shape[-1] = SW16.shape[1]

    SG16 = SG16_3D.reshape(-1, SG16_3D.shape[-1])
    SG16T = SG16.t().contiguous()
    
    X8 = X8_3D.reshape(-1, X8_3D.shape[-1])
    SX16 = SX16_3D.reshape(-1, SX16_3D.shape[-1])
    G8 = G8_3D.reshape(-1, G8_3D.shape[-1])

    GX8, GSX16 = BQC.igemm_output_int_quantize_stochastic(G8, W8, SG16T, SW16, G8.shape[0], W8.shape[1], G8.shape[1])

    GW = BQC.igemm_output_fp_no_quantize(G8T, X8, SG16, SX16, G8T.shape[0], X8.shape[1], G8T.shape[1])

    if dgrad_quantize:
        GX8_3D = GX8.reshape(GX8_3D_shape)
        GSX16_3D = GSX16.reshape(GSX16_3D_shape)

        return GX8_3D, GSX16_3D, GW
    else:
        GX16 = GX8.reshape(GX8.shape[0] // B, B, GX8.shape[1] // B, B) * GSX16.reshape(GX8.shape[0] // B, GX8.shape[1] // B).unsqueeze(1).unsqueeze(3)
        GX16 = GX16.reshape(GX8.shape).reshape(GX8_3D_shape)
        return GX16, GW