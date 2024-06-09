from huggingface_hub import HfFileSystem, hf_hub_download
from .models import EncodecMAE
from ginpipe.core import gin_configure_externals
import gin
import torch
from pathlib import Path
import copy

def traverse_dir(dir, result, fs):
    for x in fs.ls(dir, refresh=True):
        if x['name'].endswith('.pt'):
            result.append('/'.join(x['name'].split('/')[2:]))
        else:
            if x['type'] == 'dir':
                traverse_dir(x['name'], result, fs)

def get_available_models():
    fs = HfFileSystem()
    available_models = []
    traverse_dir('lpepino/encodecmae-v2', available_models, fs)
    #available_models = [x['name'].split('/')[-1] for x in fs.ls('lpepino/encodecmae-pretrained/upstreams', refresh=True)]
    return [x.split('.')[0] for x in available_models]

@gin.configurable
def get_model(model, processor):
    return model, processor

def load_model(model, mode='eval',device='cuda:0'):
    #Get model files
    config_str = gin.config_str()
    registers = copy.deepcopy(gin.config._REGISTRY)
    gin.clear_config()
    registry_clear_keys = [k for k,v in gin.config._REGISTRY.items() if k not in ['gin.macro', 'gin.constant', 'gin.singleton', 'ginpipe.core.execute_pipeline', 'encodecmae.hub.get_model']]
    for k in registry_clear_keys:
        gin.config._REGISTRY.pop(k)
    available_models = get_available_models()
    if model in available_models:
        model_file = hf_hub_download(repo_id='lpepino/encodecmae-v2', filename='{}.pt'.format(model))
    else:
        raise Exception("Available models are: {}".format(available_models))

    model_state = torch.load(model_file, map_location='cpu')

    flag = {'module_list_str': model_state['imports']}
    gin_configure_externals(flag)
    gin.parse_config(model_state['config'])
    model, processor = get_model()
    model = model()
    processor = processor()
    model.load_state_dict(model_state['state_dict'])
    gin.clear_config()
    gin.config._REGISTRY.clear()
    gin.config._REGISTRY = registers
    gin.parse_config(config_str)
    if mode=='eval':
        model.eval()
    model.to(device)
    model.processor = processor
    
    #To avoid dynamic batch problems:
    if hasattr(model.visible_encoder, 'compile'):
        model.visible_encoder.compile=False
    return model
