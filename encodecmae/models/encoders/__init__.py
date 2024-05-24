import torch.nn as nn
from encodec import EncodecModel
import encodec
import torch

class WavEncoder(torch.nn.Module):
    def __init__(self, encoder, pre_net=None, post_net=None, 
                 key_in='wav', key_out='wav_features', key_lens='wav_lens',
                 make_padding_mask=True, hop_length=None,fs=None):
        super().__init__()
        self.encoder = encoder()
        self.pre_net = pre_net() if pre_net is not None else None
        self.post_net = post_net() if post_net is not None else None
        self.key_in = key_in
        self.key_out = key_out
        self.key_wav_lens = key_lens
        self.make_padding_mask = make_padding_mask
        if hop_length is None:
            self.hop_length=self.encoder.hop_length
        else:
            self.hop_length=hop_length
        if fs is None:
            self.fs = self.encoder.fs
        else:
            self.fs = fs
    
    def forward(self,x):
        #Encode wav
        y = x[self.key_in]
        if self.pre_net is not None:
            y = self.pre_net(y)
            x[key_out+'_pre_net_output'] = y
        y = self.encoder(y)
        if self.post_net is not None:
            x[self.key_out+'_encoder_out'] = y
            y = self.post_net(y)
        x[self.key_out] = y
        #Make padding masks
        if self.make_padding_mask:
            x['features_len'] = (x['wav_lens']//self.hop_length).to(y.device)
            x['feature_padding_mask'] = x['features_len'][:,None] <= torch.arange(0,y.shape[1],device=y.device)[None,:]
            
class EncodecEncoder(torch.nn.Module):
    def __init__(self, frozen: bool = True, scale: float = 1.0, pretrained: bool = True) -> None:
        """Initialize Encodec Encoder model.

        Args:
            frozen (bool, optional): Whether the model is frozen or not. Defaults to True.
            scale (float, optional): Scaling factor. Defaults to 1.0.
            pretrained (bool, optional): Whether to load a pretrained checkpoint or train from scratch. Defaults to True.
        """
        super().__init__()
        self.model = EncodecModel.encodec_model_24khz(pretrained=pretrained).encoder
        self.hop_length = self.model.hop_length
        self.frozen = frozen
        if self.frozen:
            self.model.eval()
        else:
            self.model.train()
        self.out_dim = 128
        self.fs = 24000
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor corresponding to waveforms with shape [B, T].

        Returns:
            torch.Tensor: Output from EnCodec encoder with shape [B, T, D]
        """
        x = x.unsqueeze(1)
        if self.frozen:
            with torch.no_grad():
                y = self.model(x)
        else:
            y = self.model(x)
        y = torch.permute(y,(0,2,1))*self.scale
        return y

class SEANetEncoder(torch.nn.Module):
    def __init__(self, frozen=False, lstm=False, pretrained=True):
        super().__init__()
        self.model = encodec.modules.SEANetEncoder(lstm=lstm)
        if pretrained:
            ckpt_path = 'https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th'
            sd = torch.hub.load_state_dict_from_url(ckpt_path, map_location='cpu', check_hash=True)
            sd = {k.replace('encoder.',''):v for k,v in sd.items()}
            if not lstm:
                sd = {k.replace('.15','.14'): v for k,v in sd.items()}
            self.model.load_state_dict(sd, strict=False)
        self.hop_length = self.model.hop_length
        self.frozen = frozen
        self.fs = 24000
        if self.frozen:
            self.model.eval()
        else:
            self.model.train()
        self.out_dim = self.model.dimension

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.frozen:
            with torch.no_grad():
                y = self.model(x)
        else:
            y = self.model(x)
        y = torch.permute(y,(0,2,1))
        return y  