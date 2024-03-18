import torch.nn as nn
from encodec import EncodecModel
import torch

class EncodecEncoder(nn.Module):
    def __init__(self, frozen=True, scale=1.0):
        super().__init__()
        model = EncodecModel.encodec_model_24khz()
        self.model = model.encoder
        self.hop_length = self.model.hop_length
        self.fs = 24000
        self.frozen = frozen
        if self.frozen:
            self.model.eval()
        self.out_dim = 128
        self.scale = scale

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.frozen:
            with torch.no_grad():
                y = self.model(x)
        else:
            y = self.model(x)
        y = torch.permute(y,(0,2,1))*self.scale
        return y
