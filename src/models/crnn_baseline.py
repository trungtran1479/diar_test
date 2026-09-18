import torch
import torch.nn as nn
from typing import Tuple

class CausalConv2d(nn.Module):
    """Causal 2D convolution for streaming compatibility.
    Pads only in the past.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        
        # Time dimension is first in our 2D representation: (freq, time) or (time, freq)?
        # PyTorch Conv2d: [B, C, H, W]. Let's say H is freq, W is time.
        # Pad freq symmetrically, pad time causally (left only).
        self.pad_freq = self.kernel_size[0] // 2
        self.pad_time = self.kernel_size[1] - 1
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=(self.pad_freq, 0)
        )
        
    def forward(self, x):
        # x: [B, C, F, T]
        x = nn.functional.pad(x, (self.pad_time, 0, 0, 0))
        return self.conv(x)

class CRNNBackbone(nn.Module):
    """CRNN Backbone (Variant A1: causal GRU, A2: BiGRU).
    
    Architecture:
    - 3-layer Causal Conv2D (downsamples frequency)
    - Unidirectional GRU (A1, streaming) or Bidirectional GRU (A2, offline)
    """
    def __init__(
        self, 
        input_dim: int = 80, 
        hidden_dim: int = 256, 
        num_layers: int = 3,
        output_dim: int = 512,
        bidirectional: bool = False
    ):
        super().__init__()
        self.bidirectional = bidirectional
        
        # Conv block (Frequency downsampling)
        self.conv1 = CausalConv2d(1, 16, kernel_size=(3, 3), stride=(2, 1))
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = CausalConv2d(16, 32, kernel_size=(3, 3), stride=(2, 1))
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = CausalConv2d(32, 64, kernel_size=(3, 3), stride=(2, 1))
        self.bn3 = nn.BatchNorm2d(64)
        
        self.act = nn.ReLU()
        
        # Calculate resulting frequency dimension
        f = input_dim
        f = (f + 2 * (3 // 2) - 3) // 2 + 1  # conv1
        f = (f + 2 * (3 // 2) - 3) // 2 + 1  # conv2
        f = (f + 2 * (3 // 2) - 3) // 2 + 1  # conv3
        
        rnn_in_dim = f * 64
        
        self.rnn = nn.GRU(
            input_size=rnn_in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        rnn_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.proj = nn.Linear(rnn_out_dim, output_dim)
        
    def forward_features(self, features: torch.Tensor, feature_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, T, 80]
            feature_lens: [B]
        Returns:
            h: [B, T_enc, output_dim]
            h_lens: [B]
        """
        # [B, T, F] -> [B, 1, F, T]
        x = features.unsqueeze(1).transpose(2, 3)
        
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        
        # [B, C, F, T] -> [B, T, C*F]
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        
        # RNN
        # For a true implementation, we should use pack_padded_sequence here
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        
        # Project
        h = self.proj(out)
        
        # Length remains the same in time dimension for this CNN
        h_lens = feature_lens
        
        return h, h_lens
