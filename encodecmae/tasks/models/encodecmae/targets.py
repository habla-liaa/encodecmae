from encodec import EncodecModel
import torch
import torch.nn as nn

class EncodecQuantizer(nn.Module):
    def __init__(self, n=8, frozen=True, scale=1.0):
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
    
    def forward(self, x):
        with torch.no_grad():
            x = torch.transpose(x,1,2)
            result = self.model(x,sample_rate=75,bandwidth=self.bandwidth)
            y = result.codes
        return y