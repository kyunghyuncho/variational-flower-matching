import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import pytorch_lightning as pl
from torch.distributions import Normal

import numpy as np
import matplotlib.pyplot as plt

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback


# Define the Constraint-Coding Network
class ConstraintCodingNetwork(nn.Module):
    def __init__(self, constraint_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(constraint_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, output_dim)
        # self.fc_log_var = nn.Linear(256, output_dim)

    def forward(self, constraints):
        x = F.relu(self.fc1(constraints))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        # log_var = self.fc_log_var(x)
        log_var = torch.zeros_like(mean) # Assuming unit variance for simplicity
        return mean, log_var

# Define the Flow Matching Model
class FlowMatchingModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128) # Input: interpolated point + time
        self.t1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 256)
        self.t2 = nn.Linear(1, 256)
        self.fc3 = nn.Linear(256, 256)
        self.t3 = nn.Linear(1, 256)
        self.fc4 = nn.Linear(256, input_dim)
        self.gate = nn.Linear(256, input_dim)

    def forward(self, x_t, t):
        # Concatenate input and time
        x = F.relu(self.fc1(x_t) + self.t1(t.unsqueeze(-1)))
        x = F.relu(self.fc2(x) + self.t2(t.unsqueeze(-1)))
        x = F.relu(self.fc3(x) + self.t3(t.unsqueeze(-1)))
        g = torch.sigmoid(self.gate(x))
        v_t_predicted = g * self.fc4(x) + (1-g) * x_t
        return v_t_predicted

# Define the MNIST Dataset with Constraints (including color flip)
class MNISTWithConstraints(Dataset):
    def __init__(self, root, train=True, transform=None, flip_prob=0.5):
        self.mnist_data = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        self.num_samples = len(self.mnist_data)
        self.flip_prob = flip_prob

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, label = self.mnist_data[idx]
        img_flattened = img.flatten()
        constraints = self._get_constraints(label)

        # Randomly flip pixel values to simulate color (black/white)
        if torch.rand(1) < self.flip_prob:
            img_flattened = 1.0 - img_flattened
            constraints[2] = 1.0 # 1 represents "white" (flipped)
        else:
            constraints[2] = 0.0 # 0 represents "black" (original)

        return img_flattened, constraints

    def _get_constraints(self, label):
        constraints = torch.zeros(3)
        # Even or odd
        constraints[0] = 1 if label % 2 == 0 else 0
        # Greater than 4
        constraints[1] = 1 if label > 4 else 0
        return constraints

# Define the Data Module
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64, num_workers=4, flip_prob=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flip_prob = flip_prob
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def train_dataloader(self):
        mnist_dataset = MNISTWithConstraints(self.data_dir, train=True, transform=self.transform, flip_prob=self.flip_prob)
        return DataLoader(mnist_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        mnist_dataset = MNISTWithConstraints(self.data_dir, train=False, transform=self.transform)
        return DataLoader(mnist_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        mnist_dataset = MNISTWithConstraints(self.data_dir, train=False, transform=self.transform)
        return DataLoader(mnist_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# Define a callback to sample and log images
class SampleAndLogImagesCallback(Callback):
    def __init__(self, num_samples=16):
        super().__init__()
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            # Define the constraints for sampling
            constraints_list = [
                torch.tensor([0.,0.,0.]), # Even, > 4, Black
                torch.tensor([1.,0.,0.]), # Odd, > 4, Black
                torch.tensor([0.,1.,0.]), # Even, <= 4, Black
                torch.tensor([1.,1.,0.]), # Odd, <= 4, Black
            ]

            for i, constraints in enumerate(constraints_list):
                constraints = constraints.unsqueeze(0).repeat(self.num_samples, 1).to(pl_module.device)
                sampled_images = pl_module.sample(self.num_samples, constraints)
                self._log_images(trainer, sampled_images, f"Epoch_{trainer.current_epoch}_Samples_{i}")
        pl_module.train()

    def _log_images(self, trainer, samples, title):
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(samples[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        plt.suptitle(title)
        try:
            trainer.logger.experiment.log({title: wandb.Image(fig)})
        except Exception as e:
            print(f"Error logging images for {title} to Wandb: {e}") # Add this line
        plt.close(fig)

# Define the Flow Matching Lightning Module
class FlowMatchingLightningModule(pl.LightningModule):
    def __init__(self, 
                 input_dim=784, 
                 constraint_dim=3, 
                 ccn_output_dim=784, 
                 lr=1e-3, 
                 kl_weight=1e-4,
                 constraint_conditional=True):
        super().__init__()
        self.flow_model = FlowMatchingModel(input_dim)
        self.constraint_conditional = constraint_conditional
        if constraint_conditional:
            self.constraint_coding_network = ConstraintCodingNetwork(constraint_dim, ccn_output_dim)
        self.lr = lr
        self.kl_weight = kl_weight
        self.input_dim = input_dim
        self.normal_prior = None # Initialize as None

    def setup(self, stage=None):
        # Create the normal prior on the device of the module
        self.normal_prior = Normal(torch.zeros(self.input_dim, device=self.device),
                                   torch.ones(self.input_dim, device=self.device))

    def forward(self, x_t, t):
        return self.flow_model(x_t, t)

    def training_step(self, batch, batch_idx):
        x, constraints = batch
        batch_size = x.size(0)

        # Sample time uniformly
        t = torch.rand(batch_size, device=self.device)

        if self.constraint_conditional:
            # Get prior parameters from constraint-coding network
            prior_mean, prior_log_var = self.constraint_coding_network(constraints)
            prior_std = torch.exp(0.5 * prior_log_var)
            prior_distribution = Normal(prior_mean, prior_std)
            z = prior_distribution.sample()
        else:
            # Sample from the normal prior
            z = self.normal_prior.sample((batch_size,))

        # Conditional Gaussian path
        x_t = (1 - t.unsqueeze(-1)) * z + t.unsqueeze(-1) * x

        # Predict the velocity field
        v_t_predicted = self(x_t, t)

        # The velocity field for conditional Gaussian path
        v_t_true = x - z

        # Flow matching loss (MSE between predicted and true velocity)
        loss_fm = F.mse_loss(v_t_predicted, v_t_true)

        if self.constraint_conditional:
            # KL divergence regularization for the constraint-coding network
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(prior_distribution, self.normal_prior))
        else:
            kl_loss = torch.tensor(0.0, device=self.device)

        loss = loss_fm + self.kl_weight * kl_loss

        self.log('train_loss_fm', loss_fm)
        self.log('train_loss_kl', kl_loss)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def sample(self, num_samples, constraints):
        self.eval()
        with torch.no_grad():
            if self.constraint_conditional:
                # Get prior parameters for the given constraints
                prior_mean, prior_log_var = self.constraint_coding_network(constraints.to(self.device))
                prior_std = torch.exp(0.5 * prior_log_var)
                prior_distribution = Normal(prior_mean, prior_std)
                z = prior_distribution.sample()
            else:
                # Sample from the normal prior
                z = self.normal_prior.sample((num_samples,))
            x_t = z
            num_steps = 1000
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.ones(num_samples, device=self.device) * (1 - (i * dt))
                v_t_predicted = self(x_t, t)
                x_t = x_t - v_t_predicted * dt
            return x_t.view(-1, 1, 28, 28)

if __name__ == '__main__':
    config = {
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 1000,
        "kl_weight": 1e-3,
        "input_dim": 784,
        "constraint_dim": 3,
        "ccn_output_dim": 784,
        "flip_prob": 0.,
        "constraint_conditional": False
    }

    # Initialize wandb
    wandb.init(project="flow-matching-mnist", config=config)

    # Data Module
    data_module = MNISTDataModule(batch_size=config["batch_size"], flip_prob=config["flip_prob"])

    # Model
    model = FlowMatchingLightningModule(input_dim=config["input_dim"],
                                        constraint_dim=config["constraint_dim"],
                                        ccn_output_dim=config["ccn_output_dim"],
                                        lr=config["lr"],
                                        kl_weight=config["kl_weight"],
                                        constraint_conditional=config["constraint_conditional"])

    # Wandb Logger
    wandb_logger = WandbLogger(project="flow-matching-mnist")

    # Sample and Log Images Callback
    sample_callback = SampleAndLogImagesCallback(num_samples=16)

    # Trainer
    trainer = pl.Trainer(max_epochs=config["epochs"],
                         accelerator='auto',
                         devices=1,
                         logger=wandb_logger,
                         callbacks=[sample_callback]) # Add the callback to the trainer

    # Train the model
    trainer.fit(model, data_module)

    # Finish wandb run
    wandb.finish()