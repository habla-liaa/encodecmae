from encodecmae import load_model as load_encodecmae_model
from torch import Tensor
import torch
import sys
from huggingface_hub import hf_hub_download

def load_model(file_path, huggingface_ckpt=None):
    model = load_encodecmae_model(file_path)
    if huggingface_ckpt is not None:
        repo_id = '/'.join(huggingface_ckpt.split('/')[:2])
        filename = '/'.join(huggingface_ckpt.split('/')[2:])
        ckpt_file = hf_hub_download(repo_id=repo_id,filename=filename)
        ckpt = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])

    model.sample_rate = 24000
    model.embedding_rate=75
    model.visible_encoder.compile=False
    model.head = None

    del model.optimizer
    return model

def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
    hop_size: float = 13,
) -> Tensor:

    with torch.no_grad():
        model_device = next(model.parameters()).device
        embeddings = model.extract_features_from_array(audio)
        embeddings = torch.from_numpy(embeddings).to(model_device)
        timestamps = torch.arange(0,embeddings.shape[1])/model.embedding_rate + (0.5/model.embedding_rate)
        timestamps = torch.tile(timestamps[None,:],[embeddings.shape[0],1])

    return embeddings, timestamps.to(model_device, dtype=torch.float32)

def get_scene_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tensor:

    y, t = get_timestamp_embeddings(audio, model)
    out = torch.mean(y,axis=1)

    return out