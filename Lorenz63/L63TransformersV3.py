import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.optim as optim

# Load data from CSV file
data = pd.read_csv('lorenz_63_data.csv')

# Assume columns 'X', 'Y', 'Z' in your data
data_x = data[['X', 'Y', 'Z']].values

# Prepare sequences
M = len(data_x)
sequences = [data_x[i - 10:i] for i in range(10, M)]
targets = [data_x[i] for i in range(10, M)]

x_data = np.array(sequences)
y_data = np.array(targets)

# Split the data into training, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)


# Normalize function
def normalize(tensor):
    return (tensor - tensor.mean()) / tensor.std()


# Convert data to tensors and normalize
x_train = normalize(torch.tensor(x_train, dtype=torch.float32))
y_train = normalize(torch.tensor(y_train, dtype=torch.float32))
x_val = normalize(torch.tensor(x_val, dtype=torch.float32))
y_val = normalize(torch.tensor(y_val, dtype=torch.float32))
x_test = normalize(torch.tensor(x_test, dtype=torch.float32))
y_test = normalize(torch.tensor(y_test, dtype=torch.float32))


# Define the model
class TransformerPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.5):  # Increased dropout
        super(TransformerPredictor, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=dropout)
        self.linear_out = nn.Linear(d_model, 3)
        self.dropout = nn.Dropout(dropout)  # Added dropout

    def forward(self, src):
        src = self.dropout(src)  # Apply dropout to inputs
        output = self.transformer(src, src)
        return self.linear_out(output)


# Instantiate the model
model = TransformerPredictor(d_model=3, nhead=3, num_encoder_layers=2, num_decoder_layers=2)

# Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Updated LR scheduler

# Training parameters
epochs = 100
batch_size = 100
train_losses, val_losses = [], []

# Initialize the index set for training and validation
train_indexes = np.arange(len(x_train))
val_indexes = np.arange(len(x_val))

# Training and validation loop
for epoch in range(epochs):
    # Shuffle the indexes at the start of each epoch
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
        outputs = model(batch_x)
        loss = loss_fn(outputs[:, -1, :], batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / (len(x_train) / batch_size)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(x_val), batch_size):
            batch_indexes = val_indexes[i:i + batch_size]
            batch_x = x_val[batch_indexes]
            batch_y = y_val[batch_indexes]

            outputs = model(batch_x)
            loss = loss_fn(outputs[:, -1, :], batch_y)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / (len(x_val) / batch_size)
    val_losses.append(avg_val_loss)

    scheduler.step()  # Adjust learning rate

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

# Testing loop
model.eval()
test_losses = []
with torch.no_grad():
    for i in range(0, len(x_test), batch_size):
        batch_x = x_test[i:i + batch_size]
        batch_y = y_test[i:i + batch_size]

        outputs = model(batch_x)
        loss = loss_fn(outputs[:, -1, :], batch_y)
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
