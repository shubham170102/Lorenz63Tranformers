import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
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

# Normalize function
def normalize(tensor):
    return (tensor - tensor.mean(dim=0)) / tensor.std(dim=0)

# Define the model
class TransformerPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.5):
        super(TransformerPredictor, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=dropout)
        self.linear_out = nn.Linear(d_model, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.dropout(src)
        output = self.transformer(src, src)
        return self.linear_out(output)

# Loss function
loss_fn = nn.MSELoss()

# Training parameters
epochs = 100
batch_size = 100
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Early stopping parameters
early_stopping_patience = 10
early_stopping_delta = 0.0001

# Warmup parameters
warmup_epochs = 5

# Prepare to collect fold performances
all_fold_performances = []

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kf.split(x_data)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Split data
    x_train_fold = x_data[train_ids]
    y_train_fold = y_data[train_ids]
    x_val_fold = x_data[test_ids]
    y_val_fold = y_data[test_ids]

    # Convert data to tensors and normalize
    x_train_fold = normalize(torch.tensor(x_train_fold, dtype=torch.float32))
    y_train_fold = normalize(torch.tensor(y_train_fold, dtype=torch.float32))
    x_val_fold = normalize(torch.tensor(x_val_fold, dtype=torch.float32))
    y_val_fold = normalize(torch.tensor(y_val_fold, dtype=torch.float32))

    # Instantiate model
    model = TransformerPredictor(d_model=3, nhead=3, num_encoder_layers=2, num_decoder_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)

    # Initialize early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Store losses for plotting
    fold_train_losses = []
    fold_val_losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []

        # Shuffle training data
        perm_idx = np.random.permutation(len(x_train_fold))
        x_train_fold = x_train_fold[perm_idx]
        y_train_fold = y_train_fold[perm_idx]

        for i in range(0, len(x_train_fold), batch_size):
            batch_x = x_train_fold[i:i + batch_size]
            batch_y = y_train_fold[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs[:, -1, :], batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(x_val_fold), batch_size):
                batch_x = x_val_fold[i:i + batch_size]
                batch_y = y_val_fold[i:i + batch_size]

                outputs = model(batch_x)
                val_loss = loss_fn(outputs[:, -1, :], batch_y)
                val_losses.append(val_loss.item())

        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        fold_train_losses.append(avg_train_loss)
        fold_val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

        # Adjust learning rate
        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_step.step()

    # Save fold performance
    all_fold_performances.append({
        'fold': fold,
        'train_loss': fold_train_losses,
        'val_loss': fold_val_losses,
        'best_val_loss': best_val_loss
    })
    print(f"The best validation loss for fold {fold} was {best_val_loss:.4f}")
    print('--------------------------------')

# Save fold performances to a file
perf_df = pd.DataFrame(all_fold_performances)
perf_df.to_csv('fold_performances.csv', index=False)

# Plotting the loss curves
for perf in all_fold_performances:
    plt.figure()
    plt.plot(perf['train_loss'], label='Training Loss')
    plt.plot(perf['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curves for Fold {perf["fold"]}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'training_validation_loss_fold_{perf["fold"]}.png')
    plt.show()
