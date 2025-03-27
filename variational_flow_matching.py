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
class PosteriorUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(PosteriorUNet, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc1_norm = nn.GroupNorm(1, hidden_dim)  # GroupNorm with 1 group
        
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc2_norm = nn.GroupNorm(1, hidden_dim)
        
        self.enc3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc3_norm = nn.GroupNorm(1, hidden_dim)
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.bottleneck_norm = nn.GroupNorm(1, hidden_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.bottleneck_fc = nn.Linear(hidden_dim, hidden_dim)  # Fully connected layer
        self.bottleneck_expand = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)  # Expand back
        
        # Decoder (upsampling path)
        self.dec3 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding='same')
        self.dec3_norm = nn.GroupNorm(1, hidden_dim)
        
        self.dec2 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding='same')
        self.dec2_norm = nn.GroupNorm(1, hidden_dim)
        
        self.dec1 = nn.Conv2d(hidden_dim * 2, 2, kernel_size=kernel_size, padding='same')
        
        # Max pooling for downsampling
        self.pool = nn.MaxPool2d(2)
        
        # Upsampling operations
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # Encoder
        enc1_out = F.relu(self.enc1_norm(self.enc1(x)))
        pool1 = self.pool(enc1_out)
        
        enc2_out = F.relu(self.enc2_norm(self.enc2(pool1)))
        pool2 = self.pool(enc2_out)
        
        enc3_out = F.relu(self.enc3_norm(self.enc3(pool2)))
        pool3 = self.pool(enc3_out)
        
        # Bottleneck with global pooling
        bottleneck = F.relu(self.bottleneck_norm(self.bottleneck_conv(pool3)))
        global_features = self.global_pool(bottleneck).squeeze(-1).squeeze(-1)  # Global pooling
        global_features = F.relu(self.bottleneck_fc(global_features))  # Fully connected layer
        global_features = global_features.unsqueeze(-1).unsqueeze(-1)  # Expand dimensions
        bottleneck = self.bottleneck_expand(global_features) + bottleneck  # Combine global and local features
        
        # Decoder with skip connections
        up3 = self.up3(bottleneck)
        if up3.shape != enc3_out.shape:
            diff_h = enc3_out.size()[2] - up3.size()[2]
            diff_w = enc3_out.size()[3] - up3.size()[3]
            up3 = F.pad(up3, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2])
        skip3 = torch.cat([up3, enc3_out], dim=1)
        dec3_out = F.relu(self.dec3_norm(self.dec3(skip3)))
        
        up2 = self.up2(dec3_out)
        if up2.shape != enc2_out.shape:
            diff_h = enc2_out.size()[2] - up2.size()[2]
            diff_w = enc2_out.size()[3] - up2.size()[3]
            up2 = F.pad(up2, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2])
        skip2 = torch.cat([up2, enc2_out], dim=1)
        dec2_out = F.relu(self.dec2_norm(self.dec2(skip2)))
        
        up1 = self.up2(dec2_out)
        if up1.shape != enc1_out.shape:
            diff_h = enc1_out.size()[2] - up1.size()[2]
            diff_w = enc1_out.size()[3] - up1.size()[3]
            up1 = F.pad(up1, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2])
        skip1 = torch.cat([up1, enc1_out], dim=1)
        output = self.dec1(skip1)
        
        # Split the output into mean and logvar
        mean, logvar = torch.chunk(output, 2, dim=1)
        
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
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(VelocityFieldUNet, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc1_norm = nn.GroupNorm(1, hidden_dim)  # GroupNorm with 1 group
        self.temp_enc1 = nn.Linear(1, hidden_dim)
        
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc2_norm = nn.GroupNorm(1, hidden_dim)
        self.temp_enc2 = nn.Linear(1, hidden_dim)
        
        self.enc3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.enc3_norm = nn.GroupNorm(1, hidden_dim)
        self.temp_enc3 = nn.Linear(1, hidden_dim)
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.bottleneck_norm = nn.GroupNorm(1, hidden_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.bottleneck_fc = nn.Linear(hidden_dim, hidden_dim)  # Fully connected layer
        self.bottleneck_expand = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)  # Expand back
        
        # Decoder (upsampling path)
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
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, temperature):
        if len(temperature.shape) == 1:
            temperature = temperature.unsqueeze(-1)
        
        # Encoder
        enc1_out = F.relu(self.enc1_norm(self.enc1(x) + self.temp_enc1(temperature).unsqueeze(-1).unsqueeze(-1)))
        pool1 = self.pool(enc1_out)
        
        enc2_out = F.relu(self.enc2_norm(self.enc2(pool1) + self.temp_enc2(temperature).unsqueeze(-1).unsqueeze(-1)))
        pool2 = self.pool(enc2_out)
        
        enc3_out = F.relu(self.enc3_norm(self.enc3(pool2) + self.temp_enc3(temperature).unsqueeze(-1).unsqueeze(-1)))
        pool3 = self.pool(enc3_out)
        
        # Bottleneck with global pooling
        bottleneck = F.relu(self.bottleneck_norm(self.bottleneck_conv(pool3) + self.temp_enc3(temperature).unsqueeze(-1).unsqueeze(-1)))
        global_features = self.global_pool(bottleneck).squeeze(-1).squeeze(-1)  # Global pooling
        global_features = F.relu(self.bottleneck_fc(global_features))  # Fully connected layer
        global_features = global_features.unsqueeze(-1).unsqueeze(-1)  # Expand dimensions
        bottleneck = self.bottleneck_expand(global_features) + bottleneck  # Combine global and local features
        
        # Decoder with skip connections
        up3 = self.up3(bottleneck)
        if up3.shape != enc3_out.shape:
            diff_h = enc3_out.size()[2] - up3.size()[2]
            diff_w = enc3_out.size()[3] - up3.size()[3]
            up3 = F.pad(up3, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2])
        skip3 = torch.cat([up3, enc3_out], dim=1)
        dec3_out = F.relu(self.dec3_norm(self.dec3(skip3) + self.temp_dec3(temperature).unsqueeze(-1).unsqueeze(-1)))
        
        up2 = self.up2(dec3_out)
        if up2.shape != enc2_out.shape:
            diff_h = enc2_out.size()[2] - up2.size()[2]
            diff_w = enc2_out.size()[3] - up2.size()[3]
            up2 = F.pad(up2, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2])
        skip2 = torch.cat([up2, enc2_out], dim=1)
        dec2_out = F.relu(self.dec2_norm(self.dec2(skip2) + self.temp_dec2(temperature).unsqueeze(-1).unsqueeze(-1)))
        
        # Final upsampling and convolution
        up1 = self.up2(dec2_out)
        if up1.shape != enc1_out.shape:
            diff_h = enc1_out.size()[2] - up1.size()[2]
            diff_w = enc1_out.size()[3] - up1.size()[3]
            up1 = F.pad(up1, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2])
        skip1 = torch.cat([up1, enc1_out], dim=1)
        output = self.dec1(skip1)
        
        return output

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
    def __init__(self, batch_size=32, image_size=(32, 32)):
        super().__init__()
        self.batch_size = batch_size
        self.target_image_size = image_size

    def prepare_data(self):
        # download the data
        datasets.CelebA(root='data', split='train', download=True)
        datasets.CelebA(root='data', split='test', download=True)

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

            # let's print out some statistics about the pixel values of samples
            print(f"Mean pixel value: {samples.mean()}")
            print(f"Std pixel value: {samples.std()}")
            print(f"Min pixel value: {samples.min()}")
            print(f"Max pixel value: {samples.max()}")

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

            for i in range(self.num_samples):
                plt.subplot(num_rows, num_cols, i+1)
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
                 visualization_interval=100,
                 image_size=(28, 28)):
        super(VariationalFlowMatching, self).__init__()
        # self.posterior = PosteriorNetwork(input_dim, hidden_dim, kernel_size)
        self.posterior = PosteriorUNet(input_dim, hidden_dim, kernel_size)
        # self.velocity_field = VelocityFieldNetwork(input_dim, hidden_dim, kernel_size)
        self.velocity_field = VelocityFieldUNet(input_dim, hidden_dim, kernel_size)
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
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

        # The velocity field for conditional Gaussian path
        v_t_true = x - z

        # predict the velocity field
        v_t_pred = self(x_t, temperature.unsqueeze(-1))

        # compute the loss
        mse_loss = F.mse_loss(v_t_pred, v_t_true)

        # KL divergence from the approximater posterior to the prior.
        # the prior is standard Normal.
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # the loss is the sum of the MSE loss and KL divergence.
        loss = mse_loss + self.kl_weight * kl_div

        self.log('train_loss', loss)
        self.log('kl_div', kl_div)
        self.log('mse_loss', mse_loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def sample(self, num_samples=16, num_steps=100):
        x_t = torch.randn(num_samples, self.input_dim, self.image_size[0], self.image_size[1], device=self.device)
        
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(num_samples, device=self.device) * (1 - (i * dt))
            v_t_predicted = self(x_t, t)  # Predict the velocity at the current point

            # Predict the next step using Euler's method
            x_t_predicted = x_t + v_t_predicted * dt
            t_next = torch.ones(num_samples, device=self.device) * (1 - ((i + 1) * dt))
            v_t_next_predicted = self(x_t_predicted, t_next) # Predict the velocity at the predicted next point

            # Correct the step using the average of the two velocities
            x_t = x_t + 0.5 * (v_t_predicted + v_t_next_predicted) * dt

        return x_t

if __name__ == "__main__":
    config = {
        'batch_size': 4,
        'epochs': 100,
        'hidden_dim': 128,
        'kernel_size': 6, 
        'learning_rate': 1e-5, #1e-3,
        'kl_weight': 1,
        'visualization_interval': 100,
        'gradient_clip_val': 1.
    }

    wandb.finish()

    # initialize wandb
    wandb.init(project='variational-flow-matching')

    # initialize the data module
    # data_module = MNISTDataModule(batch_size=config['batch_size'])
    data_module = CelebADataModule(batch_size=config['batch_size'])

    # initialize the model
    model = VariationalFlowMatching(input_dim=data_module.image_size()[0], 
                                    hidden_dim=config['hidden_dim'],
                                    learning_rate=config['learning_rate'],
                                    kernel_size=config['kernel_size'],
                                    kl_weight=config['kl_weight'],
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