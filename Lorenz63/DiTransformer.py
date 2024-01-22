import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import random
import numpy as np
import scipy.io

import argparse
import os
import gflags
import sys

from sklearn.model_selection import train_test_split
# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# Load data from CSV file
data = pd.read_csv('lorenz_63_data.csv')

# Assume columns 'X', 'Y', 'Z' in your data, skip the transient state
nskip = 1000
data_x = data[['X', 'Y', 'Z']].values[nskip:]
tt = data[['Time']].values[nskip:]

# show data
fig1 = plt.figure()
for i in range(data_x.shape[1]):
    fig1.add_subplot(3, 1, i + 1)
    plt.plot(tt[:], data_x[:, i], linewidth=1)
    plt.ylabel(r'$x_{}$'.format(i + 1))
plt.xlabel('time')

# Prepare sequences
# forecast sequence length nseq, prediction steps nprd
M = len(data_x)
nseq = 20
nprd = 5
sequences = [data_x[i - nseq:i] for i in range(nseq, M)]
targets = [data_x[i - nprd + 1:i + 1] for i in range(nseq, M)]

x_data = np.array(sequences)
y_data = np.array(targets)

# Split the data into training, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)


# Normalize function
def normalize(tensor):
    return (tensor - tensor.mean()) / tensor.std()


# Convert data to tensors and normalize
# x_train = normalize(torch.tensor(x_train, dtype=torch.float32))
# y_train = normalize(torch.tensor(y_train, dtype=torch.float32))
# x_val = normalize(torch.tensor(x_val, dtype=torch.float32))
# y_val = normalize(torch.tensor(y_val, dtype=torch.float32))
# x_test = normalize(torch.tensor(x_test, dtype=torch.float32))
# y_test = normalize(torch.tensor(y_test, dtype=torch.float32))
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Define the model
class TransformerPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.1, batch_first=True):
        super(TransformerPredictor, self).__init__()
        self.linear_enc = nn.Linear(3, d_model)  # 3 for x, y, z
        self.linear_dec = nn.Linear(3, d_model)  # 3 for x, y, z
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=dropout,
                                          batch_first=batch_first)
        self.linear_out = nn.Linear(d_model, 3)  # 3 for x, y, z

    def forward(self, encoder, decoder):
        src = self.linear_enc(encoder)
        tgt = self.linear_enc(decoder)
        output = self.transformer(src, tgt)
        return self.linear_out(output)


# Instantiate the model
model = TransformerPredictor(d_model=128, nhead=8, num_encoder_layers=2, num_decoder_layers=2)
# send model to cpu or gpus
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('This code is run by {}: {} GPU(s)'.format(dev, torch.cuda.device_count()))
model.to(dev)

# Loss and optimizer
loss_fn = nn.MSELoss(reduction='mean')
cfg = {}
cfg['schedule'] = [10, 25, 50]
cfg['lr'] = 0.005
# optimizer = optim.SGD(model.parameters(), lr = cfg['lr'], momentum = .5,weight_decay = 1e-4)
optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.9, .99), weight_decay=1e-4, amsgrad=True)


# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
def adjust_learning_rate(optimizer, epoch):
    global cfg
    if epoch in cfg['schedule']:
        cfg['lr'] *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg['lr']


# Training parameters
epochs = 100
batch_size = 500
train_losses, val_losses = [], []

# Initialize the index set for training and validation
train_indexes = np.arange(len(x_train))
val_indexes = np.arange(len(x_val))

# Training and validation loop
for epoch in range(epochs):
    # Shuffle the indexes at the start of each epoch
    adjust_learning_rate(optimizer, epoch)
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    # Training loop
    model.train()
    total_loss = 0.0
    for i in range(0, len(x_train), batch_size):
        batch_indexes = train_indexes[i:i + batch_size]
        batch_x = x_train[batch_indexes]
        batch_y = y_train[batch_indexes]

        optimizer.zero_grad()
        outputs = model(batch_x, batch_y)
        loss = loss_fn(outputs[:, -nprd:, :], batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / (len(x_train) // batch_size)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(x_val), batch_size):
            batch_indexes = val_indexes[i:i + batch_size]
            batch_x = x_val[batch_indexes]
            batch_y = y_val[batch_indexes]

            outputs = model(batch_x, batch_y)
            loss = loss_fn(outputs[:, -nprd:, :], batch_y)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / (len(x_val) // batch_size)
    val_losses.append(avg_val_loss)

    # scheduler.step(avg_val_loss)

    print(
        f"Epoch {epoch + 1}/{epochs}, lr: {cfg['lr']}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

# Testing loop
model.eval()
test_losses = []
with torch.no_grad():
    for i in range(0, len(x_test), batch_size):
        batch_x = x_test[i:i + batch_size]
        batch_y = y_test[i:i + batch_size]

        outputs = model(batch_x, batch_y)
        loss = loss_fn(outputs[:, -nprd:, :], batch_y)
        test_losses.append(loss.item())

avg_test_loss = sum(test_losses) / len(test_losses)
print(f"Test Loss: {avg_test_loss}")

# Plotting
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_validation_loss_plot.png", dpi=300)
plt.show()