import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datetime import datetime
import gin
import librosa

class EncodecMAE(pl.LightningModule):
    def __init__(self, 
                 wav_encoder,
                 target_encoder,
                 visible_encoder,
                 decoder,
                 positional_encoder,
                 masker,
                 head,
                 optimizer,
                 lr_scheduler=None,
                 masked_weight=0.9,
                 quantizer_weights=None,
                 feature_dropout=0.0,
                 n_extra_targets=0
                 ):
        super().__init__()
        self.wav_encoder = wav_encoder()
        self.target_encoder = target_encoder()
        self.masker = masker()
        self.visible_encoder = visible_encoder()
        self.feat_projector = torch.nn.Linear(self.wav_encoder.out_dim, self.visible_encoder.model_dim)
        self.decoder_projector = torch.nn.Linear(self.visible_encoder.model_dim, self.visible_encoder.model_dim)
        self.positional_encoder = positional_encoder()
        self.head = head()
        self.decoder = decoder()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.masked_weight = masked_weight
        if quantizer_weights is None:
            self.quantizer_weights=torch.tensor([1.0/self.head.num_streams]*self.head.num_streams)
        else:
            self.quantizer_weights=torch.tensor(quantizer_weights)
        self.feature_dropout = torch.nn.Dropout1d(feature_dropout)
        self.initialize_weights()
        self.n_extra_targets=n_extra_targets
        if self.n_extra_targets > 0:
            self.extra_head = head(num_streams=n_extra_targets)
        self.opt_state=None

    def encode_wav(self, x):
        x['wav_features'] = self.wav_encoder(x['wav'])
        x['projected_wav_features'] = self.feat_projector(x['wav_features'])
        if hasattr(self,'feature_dropout'):
            x['projected_wav_features'] = self.feature_dropout(x['projected_wav_features'])
        if self.positional_encoder is not None:
            x['projected_wav_features'] = self.positional_encoder(x['projected_wav_features'])
        padding_mask = torch.zeros((x['projected_wav_features'].shape[0],x['projected_wav_features'].shape[1]), dtype=bool, device=x['projected_wav_features'].device)
        x['features_len'] = x['wav_lens']//self.wav_encoder.model.hop_length
        for i, l in enumerate(x['features_len']):
            padding_mask[i,l:]=1
        x['feature_padding_mask'] = padding_mask
        return x

    def make_targets(self, x):
        x['targets'] = self.target_encoder(x['wav_features'])
        if 'extra_targets' in x:
            if x['extra_targets'].ndim==2:
                x['extra_targets'] = x['extra_targets'].unsqueeze(0)
            if x['extra_targets'].shape[-1] > x['targets'].shape[-1]:
                x['extra_targets'] = x['extra_targets'][:,:,:x['targets'].shape[-1]]
            x['targets'] = torch.cat([x['targets'], x['extra_targets']], dim=0)

    def mask(self, x, ignore_mask=False):
        if (self.training) and (not ignore_mask):
            x['non_visible_mask'], x['visible_mask'], x['visible_tokens'], x['visible_lens'], x['visible_padding_mask'] = self.masker.mask(x['projected_wav_features'], xlens=x['wav_lens']//self.wav_encoder.model.hop_length)
        else:
            x['visible_tokens'] = x['projected_wav_features']
            x['visible_mask'] = torch.ones((x['projected_wav_features'].shape[0],x['projected_wav_features'].shape[1])).to(dtype=torch.bool, device=x['visible_tokens'].device)
            x['non_visible_mask'] = torch.zeros((x['projected_wav_features'].shape[0],x['projected_wav_features'].shape[1])).to(dtype=torch.bool, device=x['visible_tokens'].device)
            x['visible_padding_mask'] = x['feature_padding_mask']
            x['visible_lens'] = x['features_len']

    def encode_visible(self, x):
        x['visible_embeddings'] = self.visible_encoder(x['visible_tokens'], padding_mask=x['visible_padding_mask'])

    def decode(self, x):
        x['decoder_in'] = self.decoder_projector(x['visible_embeddings'])
        x['decoder_in'] = self.masker.unmask(x['decoder_in'], x['non_visible_mask'], x['feature_padding_mask'],x['visible_padding_mask'])
        if self.positional_encoder is not None:
            x['decoder_in'] = self.positional_encoder(x['decoder_in'])
        x['decoder_out'] = self.decoder(x['decoder_in'], padding_mask=x['feature_padding_mask'])

    def predict_tokens(self,x, key_in='decoder_out'):
        if self.head is not None:
            x['predicted_tokens'] = self.head(x[key_in], x['features_len'])
            if self.n_extra_targets>0:
                extra_preds = self.extra_head(x[key_in], x['features_len'])
                x['predicted_tokens'] = torch.cat([x['predicted_tokens'],extra_preds],axis=2)

    def forward(self, x):
        self.encode_wav(x)
        self.mask(x)
        self.encode_visible(x)
        self.decode(x)
        self.make_targets(x)
        self.predict_tokens(x)

        return x

    def forward_finetune(self,x):
        self.encode_wav(x)
        self.mask(x)
        self.encode_visible(x)
        self.predict_tokens(x, key_in='visible_embeddings')

        return x
    
    def extract_activations(self,x, detach=True):
        if detach:
            with torch.no_grad():
                self.encode_wav(x)
                self.mask(x,ignore_mask=True)
                x['visible_encoder_activations'] = self.visible_encoder.get_activations(x['visible_tokens'], padding_mask=x['visible_padding_mask'])
        return x

    def calculate_loss(self,x):
        all_losses = {}
        loss = F.cross_entropy(torch.permute(x['predicted_tokens'],(0,3,1,2)), torch.permute(x['targets'],(1,2,0)), reduction='none')
        if 'quantizer_loss_weights' not in x:
            x['quantizer_loss_weights'] = torch.tensor([[1.0]], device=loss.device, dtype=loss.dtype)
        loss = loss*(x['quantizer_loss_weights'].unsqueeze(1))*(self.quantizer_weights.to(x['predicted_tokens'].device).unsqueeze(0).unsqueeze(0))
        
        masked_loss = loss*(x['non_visible_mask'].unsqueeze(-1))
        unmasked_loss = loss*(x['visible_mask'].unsqueeze(-1))
        for q in range(masked_loss.shape[-1]):
            all_losses[f'masked_loss_q{q}'] = torch.sum(masked_loss[:,:,q])/torch.sum(x['non_visible_mask'])
            all_losses[f'unmasked_loss_q{q}'] = torch.sum(unmasked_loss[:,:,q])/torch.sum(x['visible_mask'])
        
        masked_loss = torch.sum(masked_loss)/torch.sum(x['non_visible_mask'])
        unmasked_loss = torch.sum(unmasked_loss)/torch.sum(x['visible_mask'])
        all_losses['masked_loss'] = masked_loss
        all_losses['unmasked_loss'] = unmasked_loss
        loss = self.masked_weight*masked_loss + (1-self.masked_weight)*unmasked_loss
        all_losses['loss'] = loss
        all_losses['time'] = int(datetime.now().strftime('%y%m%d%H%M%S'))
        return all_losses

    def training_step(self,x, batch_idx):
        x = self(x)
        losses = self.calculate_loss(x)
        self.log_results(x,losses,'train')

        return losses['loss']

    def validation_step(self,x, batch_idx):
        x = self(x)
        losses = self.calculate_loss(x)
        self.log_results(x,losses,'val')
        
    def log_results(self,x,losses,prefix):
        self.log_dict({'{}_{}'.format(prefix,k): v for k,v in losses.items()})

    def set_optimizer_state(self, state):
        self.opt_state = state

    def configure_optimizers(self):
        opt = self.optimizer(self.trainer.model.parameters())
        #if self.opt_state is not None:
        #    from IPython import embed; embed()
        #    opt.load_state_dict(self.opt_state[0]['state'], self.opt_state[0]['param_groups'])

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

    def initialize_weights(self):
        w = self.feat_projector.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def extract_features_from_file(self, filename, chunk_size=4, start=None, end=None, layer=-1):
        self.visible_encoder.compile=False
        fs = 24000
        start = start/fs if start is not None else 0
        end = end/fs if end is not None else None
        duration = end - start if (end is not None and start is not None) else None
        x, fs = librosa.load(filename, sr=fs, offset=start, duration=duration)
        features = self.extract_features_from_array(x, chunk_size=chunk_size, layer=layer)
        if (features.ndim == 3) and (features.shape[0]==1):
            return features[0]
        else:
            return features
            
    def extract_features_from_array(self, audio, chunk_size=4, hop_size=4, return_type='numpy', layer=-1):
        chunk_size = int(chunk_size*24000)
        hop_size = int(hop_size*24000)

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, device=self.device, dtype=torch.float32)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        with torch.no_grad():
            acts = []
            for i in range(0,audio.shape[-1],hop_size):
                if layer == -1:
                    with torch.no_grad():
                        x = {'wav': audio[:,i:i+chunk_size], 'wav_lens': torch.tensor([audio[:,i:i+chunk_size].shape[1]], device=self.device)}
                        self.encode_wav(x)
                        self.mask(x,ignore_mask=True)
                        self.encode_visible(x)
                        acts.append(x['visible_embeddings'])
                else:
                    out_i = self.extract_activations({'wav': audio[:,i:i+chunk_size], 'wav_lens': torch.tensor([audio[:,i:i+chunk_size].shape[1]], device=self.device)})
                    activations = torch.stack(out_i['visible_encoder_activations']).squeeze(axis=1)
                    if layer != 'all':
                        activations = activations[layer]
                    if activations.ndim == 2:
                        activations = activations.unsqueeze(0)
                    acts.append(activations)

            xi = torch.cat(acts,axis=1)

            if return_type == 'numpy':
                return xi.detach().cpu().numpy()
            elif return_type == 'torch':
                return xi.detach()
            else:
                raise Exception('Unrecognized return type')