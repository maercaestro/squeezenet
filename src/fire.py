"""
Fire Module for Squeezenet by Abu Huzaifah Bidin
5 Mar 2025 at 2005 hrs
"""

import torch
import torch.nn as nn

class FireModule(nn.Module):
    """
    My own implementation of the Fire module as explained in the SqueezeNet paper.
    Since FireModule is designed in modular form, we can tune it to meet different architectures 
    depending on the task. The Fire module has three tunable parameters:

    1. `sq`: Number of 1x1 squeeze filters (should be smaller than exp1 + exp3).
    2. `exp1`: Number of 1x1 expand filters.
    3. `exp3`: Number of 3x3 expand filters.

    """

    def __init__(self, in_channels: int, sq: int, exp1: int, exp3: int):
        super(FireModule, self).__init__()

        # Ensure that the squeeze layer is smaller than the total expand filters
        if sq >= (exp1 + exp3):
            raise ValueError(f"Invalid FireModule configuration: "
                             f"squeeze ({sq}) must be smaller than expand ({exp1 + exp3}).")

        # Squeeze Layer (1×1 convolution)
        self.squeeze = nn.Conv2d(in_channels, sq, kernel_size=1, bias=False)
        self.squeeze_bn = nn.BatchNorm2d(sq)  # Added BatchNorm for stability
        self.squeeze_activation = nn.ReLU(inplace=True)  # Always follow with ReLU

        # Expand Layer (1×1 and 3×3 convolutions)
        self.expand1x1 = nn.Conv2d(sq, exp1, kernel_size=1, bias=False)
        self.expand3x3 = nn.Conv2d(sq, exp3, kernel_size=3, padding=1, bias=False)
        self.expand_bn1 = nn.BatchNorm2d(exp1)
        self.expand_bn3 = nn.BatchNorm2d(exp3)
        self.expand_activation = nn.ReLU(inplace=True)  # ReLU activation

        # Apply He Initialization (Kaiming Normal)
        nn.init.kaiming_normal_(self.squeeze.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.expand1x1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.expand3x3.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze_bn(self.squeeze(x)))  # Squeeze + BN + ReLU
        x1 = self.expand_activation(self.expand_bn1(self.expand1x1(x)))  # Expand (1×1)
        x3 = self.expand_activation(self.expand_bn3(self.expand3x3(x)))  # Expand (3×3)
        return torch.cat([x1, x3], dim=1)  # Concatenate along channel dim
