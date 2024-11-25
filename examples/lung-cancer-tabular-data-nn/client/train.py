import os
import sys

import torch
import torch.optim as optim
from data import load_data
from model import load_parameters, save_parameters
from torch.utils.data import DataLoader, TensorDataset
from fedn.utils.helpers.helpers import save_metadata


def train(in_model_path, out_model_path, data_path=None, batch_size=10, epochs=1):
    
    # Load data
    X_train, y_train = load_data(data_path)

    train_dataset = TensorDataset(X_train , y_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load model
    model = load_parameters(in_model_path)

    criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    num_epochs = epochs

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
        
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        # Print loss every 10 epochs
        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')   

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(X_train),
        "batch_size": batch_size,
        "epochs": epochs,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
