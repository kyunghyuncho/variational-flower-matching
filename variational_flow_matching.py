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
# This network is a convolutional neural network.
# The input is a 28x28 image, and the output is a 28x28 image.
# The output should be the mean and log variance of the posterior distribution.
class PosteriorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PosteriorNetwork, self).__init__()
        # the first convolutional layer should maintain the size of the image 
        # but increase the channel size to hidden_dim.
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        # we repeat this for 4 times.
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        # the final layer should reduce the channel size to 2.
        self.conv5 = nn.Conv2d(in_channels=hidden_dim, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        # the output should be the mean and log variance of the posterior distribution.
        mean, logvar = torch.chunk(x, 2, dim=1)

        return mean, logvar

# Define the velocity field network.
# This network is a convolutional neural network, just like the posterior network.
# But, this network takes input the temperature t as well.
# The output should be the velocity field.
class VelocityFieldNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VelocityFieldNetwork, self).__init__()
        # the first convolutional layer should maintain the size of the image 
        # but increase the channel size to hidden_dim.
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        # for each convolutional layer, we have a temperature projection layer.
        self.temp1 = nn.Linear(1, hidden_dim)
        # we repeat this for 4 times.
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.temp2 = nn.Linear(1, hidden_dim)
        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.temp3 = nn.Linear(1, hidden_dim)
        self.conv4 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.temp4 = nn.Linear(1, hidden_dim)
        # the final layer should reduce the channel size to input_dim.
        self.conv5 = nn.Conv2d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=3, padding=1)

    def forward(self, x, temperature):
        if len(temperature.shape) == 1:
            temperature = temperature.unsqueeze(-1)

        x = F.relu(self.conv1(x) + self.temp1(temperature).unsqueeze(-1).unsqueeze(-1))
        x = F.relu(self.conv2(x) + self.temp2(temperature).unsqueeze(-1).unsqueeze(-1))
        x = F.relu(self.conv3(x) + self.temp3(temperature).unsqueeze(-1).unsqueeze(-1))
        x = F.relu(self.conv4(x) + self.temp4(temperature).unsqueeze(-1).unsqueeze(-1))
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
    
# Define a callback to sample and log images
class SampleAndLogImagesCallback(Callback):
    def __init__(self, num_samples=16):
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            samples = pl_module.sample(num_samples=self.num_samples)
            samples = samples.squeeze().cpu().numpy()
            samples = np.clip(samples, 0, 1)
            samples = np.concatenate(samples, axis=1)

            fig = plt.figure()
            plt.imshow(samples, cmap='gray')
            plt.axis('off')
            plt.tight_layout()

            # Log the samples to wandb
            trainer.logger.experiment.log({f"Epoch_{trainer.current_epoch}_samples": [wandb.Image(fig)]})
        pl_module.train()
    
# Define the variational flow matching lightning module.
class VariationalFlowMatching(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, learning_rate, image_size=(28, 28)):
        super(VariationalFlowMatching, self).__init__()
        self.posterior = PosteriorNetwork(input_dim, hidden_dim)
        self.velocity_field = VelocityFieldNetwork(input_dim, hidden_dim)
        self.learning_rate = learning_rate
        self.image_size = image_size

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
        loss = F.mse_loss(v_t_pred, v_t_true)

        # KL divergence from the approximater posterior to the prior.
        # the prior is standard Normal.
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # the loss is the sum of the MSE loss and KL divergence.
        loss = loss + kl_div

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def sample(self, num_samples=16, num_steps=100):
        x_t = torch.randn(num_samples, 1, self.image_size[0], self.image_size[1], device=self.device)
        
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
        'batch_size': 128,
        'epochs': 10,
    }

    # initialize wandb
    wandb.init(project='variational-flow-matching')

    # initialize the data module
    data_module = MNISTDataModule(batch_size=config['batch_size'])

    # initialize the model
    model = VariationalFlowMatching(input_dim=1, hidden_dim=32, learning_rate=1e-3)

    # Sample and Log Images Callback
    sample_callback = SampleAndLogImagesCallback(num_samples=16)

    # initialize the trainer
    trainer = pl.Trainer(max_epochs=config['epochs'], 
                         accelerator='auto', 
                         devices=1,
                         logger=WandbLogger(project='variational-flow-matching'),
                         callbacks=[sample_callback])
                         
    # Train the model
    trainer.fit(model, data_module)

    # Finish wandb run
    wandb.finish()