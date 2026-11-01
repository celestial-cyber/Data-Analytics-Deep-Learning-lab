# Generative Adversarial Network (GAN) is a deep learning model
# used to generate new data similar to the training data.
# It consists of two networks:
# Generator: Creates fake data using random noise.
# Discriminator: Identifies whether data is real or fake.
# Both networks compete with each other during training.
# This program generates data points resembling a sine wave.



# Step 1: Start the program.
# Step 2: Import the required libraries.
# Step 3: Set the random seed.
# Step 4: Generate 1024 training data points.
# Step 5: Define Generator and Discriminator networks.
# Step 6: Initialize loss function and Adam optimizers.
# Step 7: Train both networks for 300 epochs.
# Step 8: Generate new data using random noise.
# Step 9: Display the generated data.
# Step 10: Stop the program.


# Import libraries
import torch
from torch import nn
import math
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(111)

# Generate training data
data = torch.zeros(1024, 2)
data[:, 0] = 2 * math.pi * torch.rand(1024)
data[:, 1] = torch.sin(data[:, 0])

# Display original data
plt.plot(data[:, 0], data[:, 1], ".")
plt.show()

# Create DataLoader
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

# Define Generator and Discriminator
G = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(),
    nn.Linear(16, 32), nn.ReLU(),
    nn.Linear(32, 2)
)

D = nn.Sequential(
    nn.Linear(2, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 1), nn.Sigmoid()
)

# Initialize loss and optimizers
loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.001)
opt_D = torch.optim.Adam(D.parameters(), lr=0.001)

# Train GAN
for epoch in range(300):
    for real in loader:
        size = real.size(0)

        # Train Discriminator
        noise = torch.randn(size, 2)
        fake = G(noise).detach()

        real_labels = torch.ones(size, 1)
        fake_labels = torch.zeros(size, 1)

        opt_D.zero_grad()
        loss_D = loss_fn(D(real), real_labels) + loss_fn(D(fake), fake_labels)
        loss_D.backward()
        opt_D.step()

        # Train Generator
        noise = torch.randn(size, 2)
        opt_G.zero_grad()
        loss_G = loss_fn(D(G(noise)), real_labels)
        loss_G.backward()
        opt_G.step()

    # Display loss
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} Loss D.: {loss_D.item():.4f}")
        print(f"Epoch: {epoch} Loss G.: {loss_G.item():.4f}")

# Generate new samples
with torch.no_grad():
    samples = G(torch.randn(100, 2))

# Display generated data
plt.plot(samples[:, 0], samples[:, 1], ".")
plt.show()


# GAN          : Generative Adversarial Network.
# Generator    : Creates fake data from random noise.
# Discriminator: Distinguishes real data from fake data.
# Noise        : Random input given to the Generator.
# ReLU         : Activation function used in hidden layers.
# Sigmoid      : Produces output between 0 and 1.
# BCELoss      : Measures classification error.
# Adam         : Optimizer used to update network weights.
# Epoch        : One complete training cycle.


# Epoch: 0   Loss D.: (training loss)
# Epoch: 0   Loss G.: (training loss)
# Epoch: 10  Loss D.: (training loss)
# Epoch: 10  Loss G.: (training loss)
# ...
# Epoch: 290 Loss D.: (training loss)
# Epoch: 290 Loss G.: (training loss)

# Output 1: Original sine-wave data plot.
# Output 2: Generated data plot containing 100 points.


# Thus, a Generative Adversarial Network (GAN) was
# successfully implemented using Generator and Discriminator
# networks to generate new data samples.