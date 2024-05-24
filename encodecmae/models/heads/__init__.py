import torch.nn as nn
import torch

class FrameLevelClassificationHead(nn.Module):
    def __init__(self, model_dim: int, num_tokens: int, num_streams: int = 8, key_in: str = 'decoder_out', key_out: str = 'predicted_tokens') -> None:
        """Initialize FrameLevelClassificationHead.

        Args:
            model_dim (int): Dimensionality of the input.
            num_tokens (int): Number of tokens.
            num_streams (int, optional): Number of streams. Defaults to 8.
            key_in (str, optional): Key for input data. Defaults to 'decoder_out'.
            key_out (str, optional): Key for output data. Defaults to 'predicted_tokens'.
        """
        super().__init__()
        self.layer = nn.Linear(model_dim,num_tokens*num_streams)
        self.num_tokens = num_tokens
        self.num_streams = num_streams
        self.ar_sampling = False
        self.key_in=key_in
        self.key_out=key_out

    def forward(self, x: dict) -> dict:
        """Forward pass.

        Args:
            x (dict): Input data dictionary.

        Returns:
            dict: Output data dictionary.
        """
        xin = x[self.key_in]
        probs = self.layer(xin).view(xin.shape[0],xin.shape[1],self.num_streams,self.num_tokens)
        x[self.key_out] = probs
        return x

class SegmentLevelClassificationHead(nn.Module):
    def __init__(self, model_dim: int, num_classes: int, num_streams: int) -> None:
        """Initialize SegmentLevelClassificationHead.

        Args:
            model_dim (int): Dimensionality of the input.
            num_classes (int): Number of classes.
            num_streams (int): Number of streams.
        """
        super().__init__()
        self.layer = nn.Linear(model_dim,num_classes*num_streams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        probs = self.layer(torch.mean(x,axis=1))
        return probs