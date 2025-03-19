import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pytorch_lightning as pl
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

# Define the Constraint-Coding Network
class ConstraintCodingNetwork(nn.Module):
    def __init__(self, constraint_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(constraint_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc_mean = nn.Linear(256, output_dim)
        self.fc_log_var = nn.Linear(256, output_dim)

    def forward(self, constraints):
        x = F.relu(self.fc1(constraints))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var

# Define the Flow Matching Model
class FlowMatchingModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, 128) # Input: interpolated point + time
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, input_dim)

    def forward(self, x_t, t):
        # Concatenate input and time
        input_tensor = torch.cat([x_t, t.unsqueeze(-1)], dim=-1)
        x = F.relu(self.fc1(input_tensor))
        x = F.relu(self.fc2(x))
        v_t_predicted = self.fc3(x)
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
        self.transform = transforms.ToTensor()

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

# Define the Flow Matching Lightning Module
class FlowMatchingLightningModule(pl.LightningModule):
    def __init__(self, input_dim=784, constraint_dim=3, ccn_output_dim=784, lr=1e-3, kl_weight=1e-4):
        super().__init__()
        self.flow_model = FlowMatchingModel(input_dim)
        self.constraint_coding_network = ConstraintCodingNetwork(constraint_dim, ccn_output_dim)
        self.lr = lr
        self.kl_weight = kl_weight
        self.input_dim = input_dim
        self.normal_prior = Normal(torch.zeros(ccn_output_dim), torch.ones(ccn_output_dim))

    def forward(self, x_t, t):
        return self.flow_model(x_t, t)

    def training_step(self, batch, batch_idx):
        x, constraints = batch
        batch_size = x.size(0)

        # Sample time uniformly
        t = torch.rand(batch_size, device=self.device)

        # Get prior parameters from constraint-coding network
        prior_mean, prior_log_var = self.constraint_coding_network(constraints)
        prior_std = torch.exp(0.5 * prior_log_var)
        prior_distribution = Normal(prior_mean, prior_std)
        z = prior_distribution.sample()

        # Conditional Gaussian path
        x_t = (1 - t.unsqueeze(-1)) * z + t.unsqueeze(-1) * x

        # Predict the velocity field
        v_t_predicted = self(x_t, t)

        # True velocity field for conditional Gaussian path
        v_t_true = x - z

        # Flow matching loss (MSE between predicted and true velocity)
        loss_fm = F.mse_loss(v_t_predicted, v_t_true)

        # KL divergence regularization for the constraint-coding network
        kl_loss = torch.mean(torch.distributions.kl.kl_divergence(prior_distribution, self.normal_prior))
        loss = loss_fm + self.kl_weight * kl_loss

        self.log('train_loss_fm', loss_fm)
        self.log('train_loss_kl', kl_loss)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def sample(self, num_samples, constraints):
        self.eval()
        with torch.no_grad():
            # Get prior parameters for the given constraints
            prior_mean, prior_log_var = self.constraint_coding_network(constraints.to(self.device))
            prior_std = torch.exp(0.5 * prior_log_var)
            prior_distribution = Normal(prior_mean, prior_std)
            z = prior_distribution.sample()
            x_t = z
            num_steps = 100
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.ones(num_samples, device=self.device) * (1 - (i * dt))
                v_t_predicted = self(x_t, t)
                x_t = x_t - v_t_predicted * dt
            return x_t.view(-1, 1, 28, 28)

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64
    lr = 1e-3
    epochs = 2
    kl_weight = 1e-4
    input_dim = 784
    constraint_dim = 3
    ccn_output_dim = 784
    flip_prob = 0.5

    # Data Module
    data_module = MNISTDataModule(batch_size=batch_size, flip_prob=flip_prob)

    # Model
    model = FlowMatchingLightningModule(input_dim=input_dim, constraint_dim=constraint_dim, ccn_output_dim=ccn_output_dim, lr=lr, kl_weight=kl_weight)

    # Trainer
    trainer = pl.Trainer(max_epochs=epochs, 
                                  accelerator='auto', 
                                  devices=1)

    # Train the model
    trainer.fit(model, data_module)

    # Sampling (example: sample digits that are even and greater than 4)
    num_samples = 16
    even_gt_4_black_constraint = torch.zeros(3)
    even_gt_4_black_constraint[0] = 1.0 # Even
    even_gt_4_black_constraint[1] = 1.0 # > 4
    even_gt_4_black_constraint[2] = 0.0 # Black
    even_gt_4_black_constraint = even_gt_4_black_constraint.unsqueeze(0).repeat(num_samples, 1)
    sampled_black_images = model.sample(num_samples, even_gt_4_black_constraint)

    even_le_4_white_constraint = torch.zeros(3)
    even_le_4_white_constraint[0] = 1.0 # Odd
    even_le_4_white_constraint[1] = 0. # <= 4
    even_le_4_white_constraint[2] = 1.0 # White
    even_le_4_white_constraint = even_le_4_white_constraint.unsqueeze(0).repeat(num_samples, 1)
    sampled_white_images = model.sample(num_samples, even_le_4_white_constraint)

    # Visualize the sampled images
    def visualize_samples(samples, title):
        _, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(samples[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        plt.suptitle(title)
        plt.show()
        plt.savefig(title + '.png')

    visualize_samples(sampled_black_images, "Sampled Even, > 4, Black Digits")
    visualize_samples(sampled_white_images, "Sampled Even, > 4, White Digits")