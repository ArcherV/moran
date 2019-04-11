import torch.nn as nn
from models.morn import MORN
from models.asrn_res import ASRN


class MORAN(nn.Module):
    def __init__(self, nc, nclass, nh, targetH, targetW,
                 inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True, log=None):
        super(MORAN, self).__init__()
        self.MORN = MORN(nc, targetH, targetW, inputDataType, maxBatch, CUDA, log)
        self.ASRN = ASRN(targetH, nc, nclass, nh, CUDA)

    def forward(self, x, length, text, text_rev, test=False, debug=False, steps=None):
        x_rectified = self.MORN(x, test, debug=debug, steps=steps)
        preds = self.ASRN(x_rectified, length, text, text_rev, test)
        return preds
