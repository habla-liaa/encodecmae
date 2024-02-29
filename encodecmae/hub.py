from huggingface_hub import hf_hub_download
from .tasks.models import EncodecMAE
from ginpipe.core import gin_configure_externals
import gin
import torch

models = ['base', 'small', 'base-st', 'large', 'large-st']
models = {k: 'lpepino/encodecmae-{}'.format(k) for k in models}

@gin.configurable
def get_model(model):
    return model

def load_model(model,mode='eval',device='cuda:0'):
    #Get model files
    config_str = gin.config_str()
    gin.clear_config()
    ckpt_file = hf_hub_download(repo_id=models[model],filename='model.pt')
    config_file = hf_hub_download(repo_id=models[model],filename='config.gin')
    import_file = hf_hub_download(repo_id=models[model],filename='imports')
    flag = {'module_list': [import_file]}
    gin_configure_externals(flag)
    gin.parse_config_file(config_file)
    model = get_model()()
    ckpt = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    gin.clear_config()
    gin.parse_config(config_str)
    if mode=='eval':
        model.eval()
    model.to(device)
    return model
