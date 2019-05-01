import torch.nn as nn
from models.asrn_res import ASRN
import torch.nn.functional as F
from models.stn import get_model


class MORAN(nn.Module):
    def __init__(self, opt, nc, nclass, nh, targetH, targetW, CUDA=True):
        super(MORAN, self).__init__()
        self.ASRN = ASRN(targetH, nc, nclass, nh, CUDA)
        self.stn = get_model(opt)
        self.targetH = targetH
        self.targetW = targetW

    def forward(self, x, length, text, text_rev, test=False, debug=False, steps=None):
        x = F.upsample(x, size=(self.targetH, self.targetW), mode='bilinear')
        x_rectified = self.stn(x)
        preds = self.ASRN(x_rectified, length, text, text_rev, test)
        return preds
