from encodecmae import load_model as load_encodecmae_model
from torch import Tensor
import torch
import sys

def load_model(file_path):
    model = load_encodecmae_model(file_path)
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
        embeddings = model.extract_features_from_array(audio, return_type='torch', hop_size=1)
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