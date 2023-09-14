import torch.nn as nn
import torch

class FrameLevelClassificationHead(nn.Module):
    def __init__(self, model_dim, num_tokens, num_streams=8):
        super().__init__()
        self.layer = nn.Linear(model_dim,num_tokens*num_streams)
        self.num_tokens = num_tokens
        self.num_streams = num_streams
        self.ar_sampling = False

    def forward(self,x, feature_lens=None):
        probs = self.layer(x).view(x.shape[0],x.shape[1],self.num_streams,self.num_tokens)
        return probs

class SegmentLevelClassificationHead(nn.Module):
    def __init__(self, model_dim, num_classes,num_streams):
        super().__init__()
        self.layer = nn.Linear(model_dim,num_classes*num_streams)

    def forward(self, x):
        probs = self.layer(torch.mean(x,axis=1))
        return probs