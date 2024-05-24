from encodec import EncodecModel
import torch
import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans
import copy

class EncodecQuantizer(torch.nn.Module):
    def __init__(self, n=8, frozen=True, scale=1.0, key_in='wav_features', key_out='targets', use_encodec_encoder=False, return_only_last=False):
        super().__init__()
        model = EncodecModel.encodec_model_24khz()
        self.model = model.quantizer
        self.scale = scale
        #Modify state dict to scale quantizer weights
        sd = self.model.state_dict()
        sd = {k: v*scale if 'embed' in k else v for k,v in sd.items()}
        self.model.load_state_dict(sd)
        self.frozen = frozen
        self.bandwidth = (10*75*n)/1000
        if self.frozen:
            self.model.eval()
        self.key_in = key_in
        self.key_out = key_out
        self.return_only_last = return_only_last
        if use_encodec_encoder:
            self.encodec_encoder = model.encoder
            self.encodec_encoder.eval()
        else:
            self.encodec_encoder = None
    
    def forward(self, xin):
        x = xin[self.key_in]
        with torch.no_grad():
            if self.encodec_encoder is not None:
                x = x.unsqueeze(1)
                x = self.encodec_encoder(x)
                x = torch.transpose(x,1,2)
            x = torch.transpose(x,1,2)
            result = self.model(x,sample_rate=75,bandwidth=self.bandwidth)
            y = result.codes
        if self.return_only_last:
            y = y[-1,:,:].unsqueeze(0)
        xin[self.key_out] = y
        return xin

class IdentityTarget(torch.nn.Module):
    def __init__(self, key_in='targets', key_out='targets'):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        self.requires_model=False

    def forward(self, x):
        xin = x[self.key_in]
        if xin.ndim==2:
            xin = xin.unsqueeze(0)
        x[self.key_out] = xin