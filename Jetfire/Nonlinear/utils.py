import torch

def random_tensor_generator(BS, SL, CDIM, QB, dtype):
    qx = torch.randn(BS, SL, CDIM, dtype=dtype).cuda()
    qx = qx * 127 / qx.abs().max()

    # triton result
    _qx = qx.reshape(BS, SL // QB, QB, CDIM // QB, QB).permute(0, 1, 3, 2, 4)
    sx = torch.randn(BS, SL // QB, CDIM // QB).cuda().to(torch.float16)
    qx = _qx.permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)
    _rqx = (_qx * sx.unsqueeze(3).unsqueeze(4))
    rqx = _rqx.permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)

    qx = qx.to(torch.int8)
    return rqx, qx, sx