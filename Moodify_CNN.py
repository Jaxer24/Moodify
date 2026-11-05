import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the function that loads the saved training and validation datasets from mock_dataset.pt
from data_loader import load_datasets

# Fix the random seed for reproducibility
SEED = 42

# Seed Python and PyTorch random operations for more repeatable runs
random.seed(SEED)
torch.manual_seed(SEED)

# Define the CNN model for moodify a 2D CNN that predicts normalized valence and arousal values
class MoodCNN(nn.Module):

    def __init__(self):
        # Initializes the parent nn.Module class
        super().__init__()

        # Feature extraction layers: progressively increase channel depth (# of feature maps),
        # while reducing spatial dimensions from 128x128 to 16x16
        self.features = nn.Sequential(
             # Block 1: extract basic features
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: extract more complex features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: extract high-level features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Adaptive pooling: Averages each of the 128 feature maps down to one value per feature map
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Regressor layers: A fullly connected layer maps the pooled feature representation to the final valence-arousal predictions 
        # The regressor takes the 128-channel pooled feature map and outputs two values
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    # Define the forward pass for the model
    def forward(self, inputs):
        return self.regressor(self.features(inputs))

# Evaluate the model on a dataset. This computes validation MSE and MAE without changing model parameters
def evaluate(model, loader, device):

    # Switch the model to evaluation mode, dropout and batch normalization are disabled during evaluation
    model.eval() 

    # Counter variables to accumulate errors and sample counts across the dataset
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    # Disable gradient computation for evaluation to save memory and computation
    with torch.no_grad():
        # Loop through the validation data batch by batch
        for inputs, targets in loader:

            # Move inputs and targets to the appropriate device (GPU or CPU)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Feeds the input spectrograms through the CNN
            predictions = model(inputs)

            # Calculate the squared error for every predicted value and adds it to the total
            total_squared_error += torch.sum(
                (predictions - targets) ** 2
            ).item()

            # Calculate the absolute error for every output and adds it to the total
            total_absolute_error += torch.sum(
                torch.abs(predictions - targets)
            ).item()

            # Add the total number of values inside targets to total_samples
            total_samples += targets.numel()

    # Return dataset-wide MSE and MAE
    return (
        total_squared_error / total_samples,
        total_absolute_error / total_samples,
    )


def main():
    # Train on the GPU if one is available, otherwise use the CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the pre-generated split so training data is consistent with the
    # baseline and validation data remains held out
    train_set, validation_set = load_datasets("mock_dataset.pt")

    # Create batches of training examples
    # Shuffle training examples between epochs to prevent the model from relying on the order of the data
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
    )
    # No shuffling for validation data because the model is not learning from it 
    # Keeping the order stable also makes evaluation more reproducible
    validation_loader = DataLoader(
        validation_set,
        batch_size=32,
        shuffle=False,
    )

    # Initialize training components
    model = MoodCNN().to(device) # Creates the CNN and moves it to the selected device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Creates the Adam optimizer with learning rate 0.001
    loss_fn = nn.MSELoss() # Creates the mean squared error loss function
    best_validation_loss = float("inf") # Start the best validation loss at infinity

    # Epoch loop for training the model

    for epoch in range(1, 11): # Runs 10 training epochs
        # Switch the model to training mode, dropout and batch normalization are enabled during training
        model.train()
        # Counter variables for tracking the epoch’s training loss
        running_loss = 0.0
        batches = 0

        # Loop through every training batch
        for inputs, targets in train_loader:
            # Move the current batch to the CPU or GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Clear gradients from the previous batch
            optimizer.zero_grad()

            # Run the current batch through the CNN
            predictions = model(inputs)

            # Calculate MSE between predictions and true labels
            loss = loss_fn(predictions, targets)

            # Perform backpropagation
            loss.backward()

            # Use the gradients to update the model’s weights
            optimizer.step()

            # Track batch loss for a simple epoch-level training summary
            running_loss += loss.item()
            batches += 1

        # Calculate the average training loss across batches
        train_loss = running_loss / batches

        # Tests the current CNN on the validation dataset
        validation_mse, validation_mae = evaluate(
            model,
            validation_loader,
            device,
        )

        # Displays the training and validation results
        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Validation MSE: {validation_mse:.4f} | "
            f"Validation MAE: {validation_mae:.4f}"
        )

        # Save the model whenever validation MSE improves
        if validation_mse < best_validation_loss:
            best_validation_loss = validation_mse
            # Save only the best weights so inference can load this checkpoint
            torch.save(model.state_dict(), "best_cnn_model.pt")
            print("Saved best CNN model as 'best_cnn_model.pt'.")

    print(f"Training complete on {device}.")


if __name__ == "__main__":
    main()
