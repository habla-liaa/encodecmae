import random
import torch
import numpy as np

class TimeGapMask:
    def __init__(self, mask_amount=150, gap_size=15, mask_prop=0.5):
        self.mask_amount = mask_amount
        self.gap_size = gap_size
        self.mask_prop = mask_prop
    
    def create_mask(self,x,mask_size,mask_length,xlens):
        mask = np.zeros((x.shape[0],x.shape[1]),dtype=bool)
        ml = (xlens*self.mask_prop).to(dtype=torch.int64)
        for i in range(x.shape[0]):
            n_masked_i = 0
            start_idxs = []
            while n_masked_i < ml[i]:
                start_idx = random.randint(0,max(0,xlens[i]-mask_length))
                start_idxs.append(start_idx)
                mask[i,start_idx:min(start_idx+mask_length,xlens[i])]=1
                n_masked_i = mask[i].sum()
            if n_masked_i > ml[i]:
                valid_start_idxs = [idx for idx in start_idxs if idx < mask.shape[1]-mask_length]
                mask[i,valid_start_idxs[0]:valid_start_idxs[0]+n_masked_i-ml[i]]=0
        mask = torch.from_numpy(mask).to(x.device)
        return mask
    
    def mask(self,x,xlens):
        padding_mask = (torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)) >= (xlens.unsqueeze(1))
        mask = self.create_mask(x,self.mask_amount,self.gap_size,xlens)
        visibles = []
        for i in range(mask.shape[0]):
            visibles.append(x[i,(~mask[i] & ~padding_mask[i])])
        visible_lens = [xi.shape[0] for xi in visibles]
        maxlen = max(visible_lens)
        visible = torch.stack([torch.nn.functional.pad(xi,(0,0,0,maxlen-xi.shape[0])) for xi in visibles])
        visible_padding_mask = (torch.arange(0,visible.shape[1],device=visible.device).unsqueeze(0)) >= torch.tensor(visible_lens, device=visible.device).unsqueeze(1)

        return mask, ~mask, visible, visible_lens, visible_padding_mask
    
    def unmask(self,x,mask,feature_padding_mask,visible_padding_mask):
        unmasked = torch.zeros((mask.shape[0],mask.shape[1],x.shape[-1]), dtype=x.dtype, device=x.device)
        unmasked[(~mask & ~feature_padding_mask)] = x[~visible_padding_mask]

        return unmasked