import torch
from datetime import datetime

class EnCodecMAEClassificationLoss(torch.nn.Module):
    def __init__(self, masked_weight=0.9, quantizer_weights=None):
        super().__init__()
        self.masked_weight = masked_weight
        self.quantizer_weights = torch.tensor(quantizer_weights)

    def forward(self, x):
        all_losses = {}
        #Sometimes because of padding/different frame rates, there might be a small difference in length between input and target representations.
        #We fix it by cropping the longest vector:
        if x['predicted_tokens'].shape[1] < x['targets'].shape[-1]:
            x['targets'] = x['targets'][:,:,:x['predicted_tokens'].shape[1]]
        elif x['predicted_tokens'].shape[1] > x['targets'].shape[-1]:
            x['predicted_tokens'] = x['predicted_tokens'][:,:x['targets'].shape[-1],:,:]
            x['non_visible_mask'] = x['non_visible_mask'][:,:x['targets'].shape[-1]]
            x['visible_mask'] = x['visible_mask'][:,:x['targets'].shape[-1]]
        else:
            pass
        loss = torch.nn.functional.cross_entropy(torch.permute(x['predicted_tokens'],(0,3,1,2)), torch.permute(x['targets'],(1,2,0)).to(dtype=torch.long), reduction='none')
        if 'quantizer_loss_weights' not in x:
            x['quantizer_loss_weights'] = torch.tensor([[1.0]], device=loss.device, dtype=loss.dtype)
        loss = loss*(x['quantizer_loss_weights'].unsqueeze(1))*(self.quantizer_weights.to(x['predicted_tokens'].device).unsqueeze(0).unsqueeze(0))
        masked_loss = loss*(x['non_visible_mask'].unsqueeze(-1))
        unmasked_loss = loss*(x['visible_mask'].unsqueeze(-1))
        
        num_non_visible = max(1,torch.sum(x['non_visible_mask']))
        num_visible = max(1,torch.sum(x['visible_mask']))
        for q in range(masked_loss.shape[-1]):
            all_losses[f'masked_loss_q{q}'] = torch.sum(masked_loss[:,:,q])/num_non_visible
            all_losses[f'unmasked_loss_q{q}'] = torch.sum(unmasked_loss[:,:,q])/num_visible
        
        masked_loss = torch.sum(masked_loss)/num_non_visible
        unmasked_loss = torch.sum(unmasked_loss)/num_visible
        all_losses['masked_loss'] = masked_loss
        all_losses['unmasked_loss'] = unmasked_loss
        loss = self.masked_weight*masked_loss + (1-self.masked_weight)*unmasked_loss
        all_losses['loss'] = loss
        all_losses['time'] = int(datetime.now().strftime('%y%m%d%H%M%S'))

        return all_losses
