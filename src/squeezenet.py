import torch
import torch.nn as nn
from fire import FireModule  # Importing FireModule

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, input_size=32):
        """
        Modular SqueezeNet implementation supporting different input sizes and class numbers.
        
        Args:
        - num_classes (int): Number of output classes.
        - input_channels (int): Number of input channels (e.g., 3 for RGB, 1 for grayscale).
        - input_size (int): Input image size (e.g., 32 for CIFAR-10, 224 for ImageNet).
        """
        super(SqueezeNet, self).__init__()

        # Adaptive kernel sizes based on input size
        kernel_size = 3 if input_size <= 32 else 7
        stride = 1 if input_size <= 32 else 2
        padding = 1 if input_size <= 32 else 3

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            FireModule(64, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            FireModule(128, 32, 128, 128),
            FireModule(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            FireModule(384, 64, 256, 256),
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


#for CIFAR-10 dataset
class SqueezeNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            FireModule(64, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            FireModule(128, 32, 128, 128),
            FireModule(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            nn.MaxPool2d(kernel_size=2, stride=2),
            FireModule(384, 64, 256, 256),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
