import random
import torch
import numpy as np

class PatchoutMask(torch.nn.Module):
    def __init__(self, masker, positional_encoder):
        super().__init__()
        self.positional_encoder = positional_encoder()
        self.masker = masker()

    def mask(self, x, ignore_mask=False):
        if self.training and not ignore_mask:
            to_mask = self.positional_encoder(x['wav_features'])
            x['non_visible_mask'] = self.masker(x)
            x['visible_mask'] = ~x['non_visible_mask']

            visibles = []
            for i in range(to_mask.shape[0]):
                visibles.append(to_mask[i,(x['visible_mask'][i] & ~x['feature_padding_mask'][i])])
            visible_lens = [xi.shape[0] for xi in visibles]
            maxlen = max(visible_lens)
            visible = torch.stack([torch.nn.functional.pad(xi,(0,0,0,maxlen-xi.shape[0])) for xi in visibles])
            visible_padding_mask = (torch.arange(0,visible.shape[1],device=visible.device).unsqueeze(0)) >= torch.tensor(visible_lens, device=visible.device).unsqueeze(1)
            x['visible_tokens'] = visible
            x['visible_lens'] = visible_lens
            x['visible_padding_mask'] = visible_padding_mask
        else:
            x['visible_tokens'] = self.positional_encoder(x['wav_features'])
            x['visible_mask'] = torch.ones(x['wav_features'].shape[:2]).to(dtype=torch.bool, device=x['visible_tokens'].device)
            x['non_visible_mask'] = torch.zeros((x['wav_features'].shape[0],x['wav_features'].shape[1])).to(dtype=torch.bool, device=x['visible_tokens'].device)
            x['visible_padding_mask'] = x['feature_padding_mask']
            x['visible_lens'] = x['features_len']
        return x

    def unmask(self, x, key_in='decoder_in', key_out='decoder_in'):
        mask = x['non_visible_mask']
        xin = x[key_in]
        feature_padding_mask = x['feature_padding_mask']
        visible_padding_mask = x['visible_padding_mask']
        unmasked = torch.zeros((mask.shape[0],mask.shape[1],xin.shape[-1]), dtype=xin.dtype, device=xin.device)
        unmasked[(~mask & ~feature_padding_mask)] = xin[~visible_padding_mask]
        x[key_out] = unmasked
        return x

class TimeGapMask(torch.nn.Module):
    def __init__(self, p_mask, gap_size):
        super().__init__()
        self.p_mask = p_mask
        self.gap_size = gap_size

    def forward(self, x):
        x_lengths = x['features_len']
        mask = np.zeros((len(x_lengths),max(x_lengths)), dtype=bool)
        ml = (x_lengths*self.p_mask).to(dtype=torch.int64)
        for i in range(len(x_lengths)):
            n_masked_i = 0
            start_idxs = []
            while n_masked_i < ml[i]:
                start_idx = random.randint(0,max(0,x_lengths[i]-self.gap_size))
                start_idxs.append(start_idx)
                mask[i,start_idx:min(start_idx+self.gap_size,x_lengths[i])]=1
                n_masked_i = mask[i].sum()
            if n_masked_i > ml[i]:
                valid_start_idxs = [idx for idx in start_idxs if idx < mask.shape[1]-self.gap_size]
                mask[i,valid_start_idxs[0]:valid_start_idxs[0]+n_masked_i-ml[i]]=0
        return torch.from_numpy(mask).to(x['wav_features'].device)