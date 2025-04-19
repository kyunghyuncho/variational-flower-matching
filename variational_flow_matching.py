import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import pytorch_lightning as pl
from torch.distributions import Normal

import numpy as np
import matplotlib.pyplot as plt
import math

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback


# Define the posterior network.
# Unet based network
# Define the posterior network.
# Unet based network
class PosteriorUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, image_size=None):
        super(PosteriorUNet, self).__init__()
        
        # Calculate max possible depth based on image size
        if image_size is not None:
            # Assuming image_size is a tuple (height, width) or a single int
            if isinstance(image_size, int):
                height = width = image_size
            else:
                height, width = image_size
                
            self.max_depth = min(int(np.log2(height)), int(np.log2(width)))
        else:
            # Default to maximum 6 layers if image size not provided
            self.max_depth = 6
            
        print(f"Initializing Posterior U-Net with maximum depth: {self.max_depth}")
        
        # Encoder (downsampling path)
        self.enc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc1_norm = nn.GroupNorm(1, hidden_dim)
        
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc2_norm = nn.GroupNorm(1, hidden_dim)
        
        self.enc3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc3_norm = nn.GroupNorm(1, hidden_dim)
        
        # Additional encoder layers - will only be used if max_depth allows
        self.enc4 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=kernel_size, padding='same')
        self.enc4_norm = nn.GroupNorm(1, hidden_dim*2)
        
        self.enc5 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=kernel_size, padding='same')
        self.enc5_norm = nn.GroupNorm(1, hidden_dim*2)
        
        self.enc6 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=kernel_size, padding='same')
        self.enc6_norm = nn.GroupNorm(1, hidden_dim*2)
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(hidden_dim*2 if self.max_depth > 3 else hidden_dim, 
                                        hidden_dim*2 if self.max_depth > 3 else hidden_dim, 
                                        kernel_size=kernel_size, padding='same')
        self.bottleneck_norm = nn.GroupNorm(1, hidden_dim*2 if self.max_depth > 3 else hidden_dim)
        
        # Decoder (upsampling path)
        self.dec6 = nn.Conv2d(hidden_dim*2 * 2, hidden_dim*2, kernel_size=kernel_size, padding='same')
        self.dec6_norm = nn.GroupNorm(1, hidden_dim*2)
        
        self.dec5 = nn.Conv2d(hidden_dim*2 * 2, hidden_dim*2, kernel_size=kernel_size, padding='same')
        self.dec5_norm = nn.GroupNorm(1, hidden_dim*2)
        
        self.dec4 = nn.Conv2d(hidden_dim*2 * 2, hidden_dim, kernel_size=kernel_size, padding='same')
        self.dec4_norm = nn.GroupNorm(1, hidden_dim)
        
        self.dec3 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding='same')
        self.dec3_norm = nn.GroupNorm(1, hidden_dim)
        
        self.dec2 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding='same')
        self.dec2_norm = nn.GroupNorm(1, hidden_dim)
        
        self.dec1 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding='same')
        self.dec1_norm = nn.GroupNorm(1, hidden_dim)
        
        # Output layers for mean and log variance
        self.mean_output = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)
        self.logvar_output = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)
        
        # Max pooling for downsampling
        self.pool = nn.MaxPool2d(2)
        
        # Upsampling operations
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # Store encoder outputs for skip connections
        enc_outputs = []
        
        # Encoder path - first 3 layers are always used
        enc1_out = F.relu(self.enc1_norm(self.enc1(x)))
        enc_outputs.append(enc1_out)
        pool = self.pool(enc1_out)
        
        enc2_out = F.relu(self.enc2_norm(self.enc2(pool)))
        enc_outputs.append(enc2_out)
        pool = self.pool(enc2_out)
        
        enc3_out = F.relu(self.enc3_norm(self.enc3(pool)))
        enc_outputs.append(enc3_out)
        pool = self.pool(enc3_out)
        
        # Additional encoder layers if depth allows
        if self.max_depth > 3:
            enc4_out = F.relu(self.enc4_norm(self.enc4(pool)))
            enc_outputs.append(enc4_out)
            pool = self.pool(enc4_out)
        
            if self.max_depth > 4:
                enc5_out = F.relu(self.enc5_norm(self.enc5(pool)))
                enc_outputs.append(enc5_out)
                pool = self.pool(enc5_out)
            
                if self.max_depth > 5:
                    enc6_out = F.relu(self.enc6_norm(self.enc6(pool)))
                    enc_outputs.append(enc6_out)
                    pool = self.pool(enc6_out)
        
        # Bottleneck
        bottleneck = F.relu(self.bottleneck_norm(self.bottleneck_conv(pool)))
        
        # Decoder path
        x = bottleneck
        
        # Additional decoder layers if depth allows
        if self.max_depth > 5:
            x = self.up(x)
            # Handle potential size mismatch
            if x.shape[2:] != enc_outputs[-1].shape[2:]:
                x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, enc_outputs.pop()], dim=1)
            x = F.relu(self.dec6_norm(self.dec6(x)))
        
        if self.max_depth > 4:
            x = self.up(x)
            if x.shape[2:] != enc_outputs[-1].shape[2:]:
                x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, enc_outputs.pop()], dim=1)
            x = F.relu(self.dec5_norm(self.dec5(x)))
        
        if self.max_depth > 3:
            x = self.up(x)
            if x.shape[2:] != enc_outputs[-1].shape[2:]:
                x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, enc_outputs.pop()], dim=1)
            x = F.relu(self.dec4_norm(self.dec4(x)))
        
        # Original decoder layers - always used
        x = self.up(x)
        if x.shape[2:] != enc_outputs[-1].shape[2:]:
            x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc_outputs.pop()], dim=1)
        x = F.relu(self.dec3_norm(self.dec3(x)))
        
        x = self.up(x)
        if x.shape[2:] != enc_outputs[-1].shape[2:]:
            x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc_outputs.pop()], dim=1)
        x = F.relu(self.dec2_norm(self.dec2(x)))
        
        x = self.up(x)
        if x.shape[2:] != enc_outputs[-1].shape[2:]:
            x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc_outputs.pop()], dim=1)
        x = F.relu(self.dec1_norm(self.dec1(x)))
        
        # Output mean and log variance
        mean = self.mean_output(x)
        logvar = self.logvar_output(x)
        
        return mean, logvar
    
# This network is a convolutional neural network.
# The input is a 28x28 image, and the output is a 28x28 image.
# The output should be the mean and log variance of the posterior distribution.
class PosteriorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(PosteriorNetwork, self).__init__()
        # the first convolutional layer should maintain the size of the image 
        # but increase the channel size to hidden_dim.
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same')
        # we repeat this for 4 times.
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same')
        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same')
        self.conv4 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same')
        # the final layer should reduce the channel size to 2.
        self.conv_final = nn.Conv2d(in_channels=hidden_dim, out_channels=2, kernel_size=kernel_size, padding='same')

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x)) + x
        x = F.tanh(self.conv3(x)) + x 
        x = F.tanh(self.conv4(x)) + x
        x = self.conv_final(x)

        # the output should be the mean and log variance of the posterior distribution.
        mean, logvar = torch.chunk(x, 2, dim=1)

        return mean, logvar

# Define the velocity field network.

# Unet based implementation
class VelocityFieldUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, image_size=None):
        super(VelocityFieldUNet, self).__init__()
        
        # Calculate max possible depth based on image size
        if image_size is not None:
            # Assuming image_size is a tuple (height, width) or a single int
            if isinstance(image_size, int):
                height = width = image_size
            else:
                height, width = image_size
                
            self.max_depth = min(int(np.log2(height)), int(np.log2(width)))
        else:
            # Default to maximum 6 layers if image size not provided
            self.max_depth = 6
            
        print(f"Initializing U-Net with maximum depth: {self.max_depth}")
        
        # Encoder (downsampling path)
        self.enc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc1_norm = nn.GroupNorm(1, hidden_dim)
        self.temp_enc1 = nn.Linear(1, hidden_dim)
        
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc2_norm = nn.GroupNorm(1, hidden_dim)
        self.temp_enc2 = nn.Linear(1, hidden_dim)
        
        self.enc3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc3_norm = nn.GroupNorm(1, hidden_dim)
        self.temp_enc3 = nn.Linear(1, hidden_dim)
        
        # Additional encoder layers - will only be used if max_depth allows
        self.enc4 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=kernel_size, padding='same')
        self.enc4_norm = nn.GroupNorm(1, hidden_dim*2)
        self.temp_enc4 = nn.Linear(1, hidden_dim*2)
        
        self.enc5 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=kernel_size, padding='same')
        self.enc5_norm = nn.GroupNorm(1, hidden_dim*2)
        self.temp_enc5 = nn.Linear(1, hidden_dim*2)
        
        self.enc6 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=kernel_size, padding='same')
        self.enc6_norm = nn.GroupNorm(1, hidden_dim*2)
        self.temp_enc6 = nn.Linear(1, hidden_dim*2)
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(hidden_dim*2 if self.max_depth > 3 else hidden_dim, 
                                         hidden_dim*2 if self.max_depth > 3 else hidden_dim, 
                                         kernel_size=kernel_size, padding='same')
        self.bottleneck_norm = nn.GroupNorm(1, hidden_dim*2 if self.max_depth > 3 else hidden_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck_fc = nn.Linear(hidden_dim*2 if self.max_depth > 3 else hidden_dim, 
                                       hidden_dim*2 if self.max_depth > 3 else hidden_dim)
        self.bottleneck_expand = nn.Conv2d(hidden_dim*2 if self.max_depth > 3 else hidden_dim, 
                                          hidden_dim*2 if self.max_depth > 3 else hidden_dim, kernel_size=1)
        
        # Decoder (upsampling path) - corresponding to encoder layers
        # Will be selected dynamically based on max_depth
        self.dec6 = nn.Conv2d(hidden_dim*2 * 2, hidden_dim*2, kernel_size=kernel_size, padding='same')
        self.dec6_norm = nn.GroupNorm(1, hidden_dim*2)
        self.temp_dec6 = nn.Linear(1, hidden_dim*2)
        
        self.dec5 = nn.Conv2d(hidden_dim*2 * 2, hidden_dim*2, kernel_size=kernel_size, padding='same')
        self.dec5_norm = nn.GroupNorm(1, hidden_dim*2)
        self.temp_dec5 = nn.Linear(1, hidden_dim*2)
        
        self.dec4 = nn.Conv2d(hidden_dim*2 * 2, hidden_dim, kernel_size=kernel_size, padding='same')
        self.dec4_norm = nn.GroupNorm(1, hidden_dim)
        self.temp_dec4 = nn.Linear(1, hidden_dim)
        
        self.dec3 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding='same')
        self.dec3_norm = nn.GroupNorm(1, hidden_dim)
        self.temp_dec3 = nn.Linear(1, hidden_dim)
        
        self.dec2 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding='same')
        self.dec2_norm = nn.GroupNorm(1, hidden_dim)
        self.temp_dec2 = nn.Linear(1, hidden_dim)
        
        self.dec1 = nn.Conv2d(hidden_dim * 2, input_dim, kernel_size=kernel_size, padding='same')
        
        # Max pooling for downsampling
        self.pool = nn.MaxPool2d(2)
        
        # Upsampling operations
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, temperature):
        if len(temperature.shape) == 1:
            temperature = temperature.unsqueeze(-1)
        
        # Store encoder outputs for skip connections
        enc_outputs = []
        
        # Encoder path - first 3 layers are always used
        enc1_out = F.relu(self.enc1_norm(self.enc1(x) + self.temp_enc1(temperature).unsqueeze(-1).unsqueeze(-1)))
        enc_outputs.append(enc1_out)
        pool = self.pool(enc1_out)
        
        enc2_out = F.relu(self.enc2_norm(self.enc2(pool) + self.temp_enc2(temperature).unsqueeze(-1).unsqueeze(-1)))
        enc_outputs.append(enc2_out)
        pool = self.pool(enc2_out)
        
        enc3_out = F.relu(self.enc3_norm(self.enc3(pool) + self.temp_enc3(temperature).unsqueeze(-1).unsqueeze(-1)))
        enc_outputs.append(enc3_out)
        pool = self.pool(enc3_out)
        
        # Additional encoder layers if depth allows
        if self.max_depth > 3:
            enc4_out = F.relu(self.enc4_norm(self.enc4(pool) + self.temp_enc4(temperature).unsqueeze(-1).unsqueeze(-1)))
            enc_outputs.append(enc4_out)
            pool = self.pool(enc4_out)
        
            if self.max_depth > 4:
                enc5_out = F.relu(self.enc5_norm(self.enc5(pool) + self.temp_enc5(temperature).unsqueeze(-1).unsqueeze(-1)))
                enc_outputs.append(enc5_out)
                pool = self.pool(enc5_out)
            
                if self.max_depth > 5:
                    enc6_out = F.relu(self.enc6_norm(self.enc6(pool) + self.temp_enc6(temperature).unsqueeze(-1).unsqueeze(-1)))
                    enc_outputs.append(enc6_out)
                    pool = self.pool(enc6_out)
        
        # Bottleneck
        bottleneck = F.relu(self.bottleneck_norm(self.bottleneck_conv(pool) + 
                                               self.temp_enc3(temperature).unsqueeze(-1).unsqueeze(-1) if self.max_depth <= 3 else
                                               self.temp_enc6(temperature).unsqueeze(-1).unsqueeze(-1)))
        
        global_features = self.global_pool(bottleneck).squeeze(-1).squeeze(-1)
        global_features = F.relu(self.bottleneck_fc(global_features))
        global_features = global_features.unsqueeze(-1).unsqueeze(-1)
        bottleneck = self.bottleneck_expand(global_features) + bottleneck
        
        # Decoder path
        x = bottleneck
        
        # Additional decoder layers if depth allows
        if self.max_depth > 5:
            x = self.up(x)
            # Handle potential size mismatch
            if x.shape[2:] != enc_outputs[-1].shape[2:]:
                x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, enc_outputs.pop()], dim=1)
            x = F.relu(self.dec6_norm(self.dec6(x) + self.temp_dec6(temperature).unsqueeze(-1).unsqueeze(-1)))
        
        if self.max_depth > 4:
            x = self.up(x)
            if x.shape[2:] != enc_outputs[-1].shape[2:]:
                x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, enc_outputs.pop()], dim=1)
            x = F.relu(self.dec5_norm(self.dec5(x) + self.temp_dec5(temperature).unsqueeze(-1).unsqueeze(-1)))
        
        if self.max_depth > 3:
            x = self.up(x)
            if x.shape[2:] != enc_outputs[-1].shape[2:]:
                x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, enc_outputs.pop()], dim=1)
            x = F.relu(self.dec4_norm(self.dec4(x) + self.temp_dec4(temperature).unsqueeze(-1).unsqueeze(-1)))
        
        # Original decoder layers - always used
        x = self.up(x)
        if x.shape[2:] != enc_outputs[-1].shape[2:]:
            x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc_outputs.pop()], dim=1)
        x = F.relu(self.dec3_norm(self.dec3(x) + self.temp_dec3(temperature).unsqueeze(-1).unsqueeze(-1)))
        
        x = self.up(x)
        if x.shape[2:] != enc_outputs[-1].shape[2:]:
            x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc_outputs.pop()], dim=1)
        x = F.relu(self.dec2_norm(self.dec2(x) + self.temp_dec2(temperature).unsqueeze(-1).unsqueeze(-1)))
        
        x = self.up(x)
        if x.shape[2:] != enc_outputs[-1].shape[2:]:
            x = F.interpolate(x, size=enc_outputs[-1].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc_outputs.pop()], dim=1)
        x = self.dec1(x)
        
        return x

# This network is a convolutional neural network, just like the posterior network.
# But, this network takes input the temperature t as well.
# The output should be the velocity field.
class VelocityFieldNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(VelocityFieldNetwork, self).__init__()
        # the first convolutional layer should maintain the size of the image 
        # but increase the channel size to hidden_dim.
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same')
        # for each convolutional layer, we have a temperature projection layer.
        self.temp1 = nn.Linear(1, hidden_dim)
        # we repeat this for 4 times.
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same')
        self.temp2 = nn.Linear(1, hidden_dim)
        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same')
        self.temp3 = nn.Linear(1, hidden_dim)
        self.conv4 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same')
        self.temp4 = nn.Linear(1, hidden_dim)
        # the final layer should reduce the channel size to input_dim.
        self.conv5 = nn.Conv2d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=kernel_size, padding='same')

    def forward(self, x, temperature):
        if len(temperature.shape) == 1:
            temperature = temperature.unsqueeze(-1)

        x = F.tanh(self.conv1(x) + self.temp1(temperature).unsqueeze(-1).unsqueeze(-1))
        x = x + F.tanh(self.conv2(x) + self.temp2(temperature).unsqueeze(-1).unsqueeze(-1))
        x = x + F.tanh(self.conv3(x) + self.temp3(temperature).unsqueeze(-1).unsqueeze(-1))
        x = x + F.tanh(self.conv4(x) + self.temp4(temperature).unsqueeze(-1).unsqueeze(-1))
        x = self.conv5(x)

        return x
    
# Prior network: this is just a Normal distribution
class PriorNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PriorNetwork, self).__init__()
        self.input_dim = input_dim
        self.mean = nn.Parameter(torch.zeros(input_dim))
        self.logvar = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        return self.mean, torch.relu(self.logvar)
    
# Let's define the MNIST Data Module: i'm using pytorch lightning.
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # download the data
        datasets.MNIST(root='data', train=True, download=True)
        datasets.MNIST(root='data', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            transform = transforms.Compose([transforms.ToTensor()])
            self.train_dataset = datasets.MNIST(root='data', train=True, transform=transform)
            self.val_dataset = datasets.MNIST(root='data', train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def image_size(self):
        return (1, 28, 28)
    
# Let's define CelebA Data Module: i'm using pytorch lightning.
class CelebADataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, image_size=(28, 28)):
        super().__init__()
        self.batch_size = batch_size
        self.target_image_size = image_size

    def prepare_data(self):
        # download the data
        datasets.CelebA(root='/teamspace/s3_folders/', split='train', download=True)
        datasets.CelebA(root='/teamspace/s3_folders/', split='test', download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            transform = transforms.Compose([
                transforms.Resize(self.target_image_size),
                transforms.ToTensor()
            ])
            self.train_dataset = datasets.CelebA(root='data', split='train', transform=transform)
            self.val_dataset = datasets.CelebA(root='data', split='test', transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def image_size(self):
        return (3, *self.target_image_size)

# Let's define SVHN Data Module: i'm using pytorch lightning.
class SVHNDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, image_size=(32, 32)):
        super().__init__()
        self.batch_size = batch_size
        self.target_image_size = image_size

    def prepare_data(self):
        # download the data
        datasets.SVHN(root='data', split='train', download=True)
        datasets.SVHN(root='data', split='test', download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            transform = transforms.Compose([
                transforms.Resize(self.target_image_size),
                transforms.ToTensor()
            ])
            self.train_dataset = datasets.SVHN(root='data', split='train', transform=transform, download=True)
            self.val_dataset = datasets.SVHN(root='data', split='test', transform=transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def image_size(self):
        return (3, *self.target_image_size)

# STL10 Data Module
class STL10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, image_size=(96, 96)):
        super().__init__()
        self.batch_size = batch_size
        self.target_image_size = image_size

    def prepare_data(self):
        # Download the data
        datasets.STL10(root='data', split='train', download=True)
        datasets.STL10(root='data', split='test', download=True)

    def setup(self, stage=None):
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.target_image_size),
            transforms.ToTensor(),
        ])

        # Load train and test datasets with transforms
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.STL10(
                root='data',
                split='train',
                transform=self.transform,
                download=True
            )

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.STL10(
                root='data',
                split='test',
                transform=self.transform,
                download=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        # STL10 doesn't have a validation set, so we'll use the test set
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    @property
    def image_size(self):
        return (3, *self.target_image_size)
    
# Define a callback to sample and log images
class SampleAndLogImagesCallback(Callback):
    def __init__(self, num_samples=16):
        self.num_samples = num_samples

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs, # Output from the training_step
        batch,   # The batch data
        batch_idx: int, # Index of the batch within the current epoch
    ):
        if trainer.global_step % pl_module.visualization_interval != 0:
            return
        
        pl_module.eval()
        with torch.no_grad():
            samples = pl_module.sample(num_samples=self.num_samples)
            samples = samples.squeeze().cpu().numpy()

            # Log the samples to wandb
            trainer.logger.experiment.log({
                f"pixel_value_histogram": wandb.Histogram(samples.flatten())
            })

            samples = np.clip(samples, 0, 1)

            fig = plt.figure()

            # samples is a 4D tensor: (num_samples, num_channels, height, width).
            # we want to plot each sample in a grid.
            # we will plot np.round(np.sqrt(num_samples)) samples in each row.
            num_rows = math.ceil(np.sqrt(self.num_samples))
            num_cols = math.ceil(self.num_samples / num_rows)

            # the first subplot is the prior.
            plt.subplot(num_rows, num_cols, 1)
            if pl_module.prior.mean.shape[0] == 1:
                plt.imshow(pl_module.prior.mean.squeeze().cpu().numpy(), cmap='gray')
            else:
                plt.imshow(pl_module.prior.mean.squeeze().cpu().numpy().transpose(1,2,0))
            plt.axis('off')

            for i in range(self.num_samples-1):
                plt.subplot(num_rows, num_cols, i+2)
                if len(samples[i].shape) < 3:
                    plt.imshow(samples[i], cmap='gray')
                else:
                    plt.imshow(samples[i].transpose(1, 2, 0))
                plt.axis('off')
            plt.tight_layout()

            # Log the samples to wandb
            trainer.logger.experiment.log({
                f"Epoch_{trainer.current_epoch}_samples": [wandb.Image(fig)],
            })

            plt.close(fig)
            
        pl_module.train()
    
# Define the variational flow matching lightning module.
class VariationalFlowMatching(pl.LightningModule):
    def __init__(self, input_dim, 
                 hidden_dim, 
                 learning_rate, 
                 kernel_size=3,
                 kl_weight=1.0,
                 prior_weight=1.0,
                 visualization_interval=100,
                 image_size=(28, 28)):
        super(VariationalFlowMatching, self).__init__()
        self.prior = PriorNetwork((input_dim, image_size[0], image_size[1]))
        # self.posterior = PosteriorNetwork(input_dim, hidden_dim, kernel_size)
        self.posterior = PosteriorUNet(input_dim, hidden_dim, kernel_size, image_size)
        # self.velocity_field = VelocityFieldNetwork(input_dim, hidden_dim, kernel_size)
        self.velocity_field = VelocityFieldUNet(input_dim, hidden_dim, kernel_size, image_size)
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.prior_weight = prior_weight
        self.input_dim = input_dim
        self.image_size = image_size
        self.visualization_interval = visualization_interval

    def forward(self, x, temperature):
        # compute the velocity field
        velocity_field = self.velocity_field(x, temperature)
        return velocity_field

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # perform approximate posterior and draw samples from it.
        mean, logvar = self.posterior(x)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

        # draw a random temperature.
        temperature = torch.rand(x.shape[0], device=self.device)

        # compute the interpolated points.
        x_t = temperature[:,None,None,None] * z + (1 - temperature[:,None,None,None]) * x

        # compute the velocity field.
        v_t_true = z - x

        # predict the final outcome.
        v_t_pred = self(x_t, temperature.unsqueeze(-1))

        # compute the loss: it should be the negative log-probability of v_t_true 
        # under the Gaussian distribution with mean v_t_pred and variance 1.
        mse_loss = F.mse_loss(v_t_pred, v_t_true)

        # KL divergence from the approximater posterior to the prior.
        # the prior mean and logvar
        prior_mean, prior_logvar = self.prior(x)
        kl_div = -0.5 * torch.sum(1 + logvar 
                                  - prior_logvar 
                                  - (logvar.exp() 
                                     + (mean - prior_mean).pow(2)) / prior_logvar.exp())

        # another loss that fits the prior to the data.
        # this is the negative log-probability of x under the prior.
        prior_loss = 0.5 * torch.sum((x - prior_mean).pow(2) / prior_logvar.exp() + prior_logvar)

        # the loss is the sum of the MSE loss, KL divergence and the prior loss.
        loss = mse_loss + self.kl_weight * kl_div + self.prior_weight * prior_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('kl_div', kl_div)
        self.log('mse_loss', mse_loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def sample(self, num_samples=16, num_steps=100):
        x_t = torch.randn(num_samples, self.input_dim, self.image_size[0], self.image_size[1], device=self.device)

        prior_mean, prior_logvar = self.prior(x_t)
        z0 = prior_mean.unsqueeze(0) + torch.exp(0.5 * prior_logvar.unsqueeze(0)) * x_t
        x_t = z0
        t = torch.ones(num_samples, device=self.device)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            # compute the velocity field
            velocity_field = self.velocity_field(x_t, t)
            # update the samples
            x_t = x_t - dt * velocity_field
            # compute the temperature
            t = t - dt
            t = torch.clamp(t, 0, 1)

        return x_t

if __name__ == "__main__":
    config = {
        'batch_size': 64,
        'epochs': 100,
        'hidden_dim': 128,
        'kernel_size': 4, 
        'learning_rate': 1e-3,
        'kl_weight': 1,
        'prior_weight': 1,
        'visualization_interval': 1000,
        'gradient_clip_val': 1.
    }

    wandb.finish()

    # initialize wandb
    wandb.init(project='variational-flow-matching')

    # initialize the data module
    # data_module = MNISTDataModule(batch_size=config['batch_size'])
    data_module = SVHNDataModule(batch_size=config['batch_size'])
    # data_module = CelebADataModule(batch_size=config['batch_size'])
    # data_module = STL10DataModule(batch_size=config['batch_size'])

    # initialize the model
    model = VariationalFlowMatching(input_dim=data_module.image_size()[0], 
                                    hidden_dim=config['hidden_dim'],
                                    learning_rate=config['learning_rate'],
                                    kernel_size=config['kernel_size'],
                                    kl_weight=config['kl_weight'],
                                    prior_weight=config['prior_weight'],
                                    image_size=data_module.image_size()[1:],
                                    visualization_interval=config['visualization_interval'])

    # Sample and Log Images Callback
    sample_callback = SampleAndLogImagesCallback(num_samples=16)

    # initialize the trainer
    trainer = pl.Trainer(max_epochs=config['epochs'], 
                         accelerator='auto', 
                         devices=1,
                         logger=WandbLogger(project='variational-flow-matching'),
                         callbacks=[sample_callback],
                         gradient_clip_val=config['gradient_clip_val'])
                         
    # Train the model
    trainer.fit(model, data_module)

    # Finish wandb run
    wandb.finish()