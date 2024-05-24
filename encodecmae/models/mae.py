import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datetime import datetime
import gin
import librosa

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datetime import datetime
import gin
import librosa

import numpy as np

class EncodecMAE(pl.LightningModule):
    """EncodecMAE.

    Args:
        wav_encoder (torch.nn.Module): Module that takes a batch of waveforms (BxT) and returns a representation (BxTxDf).
        target_encoder (torch.nn.Module): Module that generates targets from unmasked inputs.
        visible_encoder (torch.nn.Module): Module that encodes the visible tokens returning visible embeddings (BxTvxD)
        decoder (torch.nn.Module): Module that takes expanded input (with mask tokens) and generate embeddings for the whole sequence (BxTxD)
        masker (torch.nn.Module): Module that takes wav_encoder outputs and mask them. Has 2 methods: mask() that masks the embeddings, and unmask() that expands visible embeddings to original length.
        head (torch.nn.Module): Module that takes decoder output and generates predictions of the targets.
        loss (torch.nn.Module): Module that receives prediction and targets and returns model losses.
        optimizer (torch.optim.Optimizer): Torch optimizer to use during training.
        lr_scheduler (torch.lr_scheduler.LRScheduler): Torch learning rate scheduler used during training.
    """
    def __init__(self, 
                 wav_encoder: torch.nn.Module,
                 target_encoder: torch.nn.Module,
                 visible_encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 masker: torch.nn.Module,
                 head: torch.nn.Module,
                 loss: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 lr_scheduler = None
                 ):
        super().__init__()
        self.wav_encoder = wav_encoder()
        self.target_encoder = target_encoder()
        self.masker = masker()
        self.visible_encoder = visible_encoder()
        self.head = head()
        self.decoder = decoder()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss() if loss is not None else None
        
        self.apply(self._init_weights)

    def forward(self, x):
        self.wav_encoder(x)
        self.masker.mask(x)
        self.visible_encoder(x)
        self.masker.unmask(x)
        self.decoder(x)
        if hasattr(self.target_encoder,'requires_model') and (self.target_encoder.requires_model):
            self.target_encoder(x, self)
        else:
            self.target_encoder(x)
        self.head(x)

        return x

    def forward_finetune(self,x):
        self.encode_wav(x)
        self.mask(x)
        self.encode_visible(x)
        self.predict_tokens(x, key_in='visible_embeddings')

        return x
    
    def extract_activations(self,x, 
                            detach=True,
                            postnorm_last_activation=True,
                            extract_decoder=False,
                            residual_branch=0):
        if detach:
            with torch.no_grad():
                self.wav_encoder(x)
                self.masker.mask(x,ignore_mask=True)
                self.visible_encoder(x, return_activations=True, padding_mask=x['visible_padding_mask'],postnorm_last_activation=postnorm_last_activation,residual_branch=residual_branch)
                if extract_decoder:
                    self.masker.unmask(x)
                    self.decoder(x, return_activations=True,postnorm_last_activation=postnorm_last_activation,residual_branch=residual_branch)
        return x

    def training_step(self,x, batch_idx):
        x = self(x)
        losses = self.loss(x)
        self.log_results(x,losses,'train')

        return losses['loss']

    def validation_step(self,x, batch_idx):
        x = self(x)
        losses = self.loss(x)
        self.log_results(x,losses,'val')
        
    def log_results(self,x,losses,prefix):
        self.log_dict({'{}_{}'.format(prefix,k): v for k,v in losses.items()})

    def set_optimizer_state(self, state):
        self.opt_state = state

    def configure_optimizers(self):
        opt = self.optimizer(self.trainer.model.parameters())
        if self.lr_scheduler is not None:
            if self.lr_scheduler.__name__ == 'SequentialLR':
                binds = gin.get_bindings('torch.optim.lr_scheduler.SequentialLR')
                lr_scheduler = self.lr_scheduler(opt, schedulers=[s(opt) for s in binds['schedulers']])
            else:
                lr_scheduler = self.lr_scheduler(opt) if self.lr_scheduler is not None else None
        else:
            lr_scheduler = None
        del self.optimizer
        del self.lr_scheduler
        opt_config = {'optimizer': opt}
        if lr_scheduler is not None:
            opt_config['lr_scheduler'] = {'scheduler': lr_scheduler,
                                          'interval': 'step',
                                          'frequency': 1}
        return opt_config

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def extract_features_from_file(self, 
                                   filename, 
                                   chunk_size=None, 
                                   overlap=0, 
                                   start=None, 
                                   end=None, 
                                   layer=-1, 
                                   extract_decoder=False, 
                                   postnorm_last_activation=True,
                                   residual_branch=0):
        fs = self.wav_encoder.fs
        start = start/fs if start is not None else None
        end = end/fs if end is not None else None
        duration = end - start if (end is not None and start is not None) else None
        x, fs = librosa.load(filename, sr=fs, offset=start, duration=duration)
        if chunk_size is None:
            chunk_size=int(fs*self.processor._processors[0].max_length)
        features = self.extract_features_from_array(x, chunk_size=chunk_size, layer=layer, overlap=overlap, extract_decoder=extract_decoder, postnorm_last_activation=postnorm_last_activation)
        if (features.ndim == 3) and (features.shape[0]==1):
            return features[0]
        else:
            return features
            
    def apply_processors(self, xin, from_idx=1):
        def to_torch(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                return torch.tensor(x)

        batch_processed = []
        for k in range(xin['wav'].shape[0]):
            xin_i = {f:v[k] for f,v in xin.items()}
            for p in self.processor._processors[1:]:
                xin_i = p(xin_i)
            batch_processed.append(xin_i)
        xin = {k: torch.stack([to_torch(b[k]) for b in batch_processed]).to(self.device, self.dtype) for k in batch_processed[0].keys()}
        return xin

    def extract_features_from_array(self, 
                                    audio, 
                                    wav_lens=None, 
                                    chunk_size=None, 
                                    overlap=0, 
                                    return_type='numpy', 
                                    layer=-1, 
                                    min_length=2048, 
                                    extract_decoder=False, 
                                    postnorm_last_activation=True,
                                    residual_branch=0):
        if chunk_size is None:
            fs = self.wav_encoder.fs
            chunk_size=int(fs*self.processor._processors[0].max_length)

        hop_size = chunk_size*(1-overlap)
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, device=self.device, dtype=torch.float32)

        if audio.ndim == 1:
            batch_size = 1
            audio = audio[None,:]
        else:
            batch_size = audio.shape[0]

        if wav_lens is None:
            wav_lens = [audio.shape[1]]*batch_size
        
        with torch.no_grad():
            acts = []
            for mi, i in enumerate(range(0,audio.shape[-1],hop_size)):
                if audio[:,i:i+chunk_size].shape[1]>min_length:
                    wav_lens_i = [min(max(wav_lens[i] - i,0),chunk_size) for i in range(batch_size)]
                    audio_i = audio[:,i:i+chunk_size]
                    xin = {'wav': audio_i.cpu().numpy(), 'wav_lens': np.array(wav_lens_i)}
                    xin = self.apply_processors(xin)
                    if extract_decoder:
                        self.decoder.key_transformer_out='decoder_transformer'
                    out_i = self.extract_activations(xin, extract_decoder=extract_decoder, postnorm_last_activation=postnorm_last_activation, residual_branch=residual_branch)
                    activations = torch.stack(out_i['visible_embeddings_activations']).squeeze(axis=1)
                    if extract_decoder:
                        decoder_acts = torch.stack(out_i['decoder_transformer_activations']).squeeze(axis=1)
                        activations = torch.cat([activations, decoder_acts],axis=0)
                    if layer != 'all':
                        activations = activations[layer]
                    if activations.ndim == 2:
                        activations = activations.unsqueeze(0)
                    acts.append(activations)
            if acts[0].ndim == 3:
                xi = torch.cat(acts,axis=1)
            elif acts[0].ndim == 4:
                xi = torch.cat(acts,axis=2)
            else:
                raise Exception('Wrong shape of activations')
            if return_type == 'numpy':
                return xi.detach().cpu().numpy()
            elif return_type == 'torch':
                return xi.detach()
            else:
                raise Exception('Unrecognized return type')
