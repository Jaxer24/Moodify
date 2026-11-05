"""Evaluate a mean-prediction baseline on the shared dataset.

This provides a simple reference score: the model predicts the training-set
average valence and arousal for every validation example.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import load_datasets


class MeanBaseline(nn.Module):
    """Always predicts the training-set average valence and arousal."""

    def __init__(self, mean_targets):
        super().__init__()
        # A buffer moves with the model across devices and is included in its
        # state, even though this baseline has no trainable parameters.
        self.register_buffer("mean_targets", mean_targets.float())

    def forward(self, inputs):
        # Repeat the same two target values once for each input in the batch.
        batch_size = inputs.size(0)
        return self.mean_targets.unsqueeze(0).expand(batch_size, -1)


def evaluate(model, loader, loss_fn, device):
    # Evaluation disables dropout/training-only behavior and avoids tracking
    # gradients, since no weights are updated in this function.
    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)

            total_squared_error += torch.sum(
                (predictions - targets) ** 2
            ).item()
            total_absolute_error += torch.sum(
                torch.abs(predictions - targets)
            ).item()
            # Count target values, not audio examples: each example has two
            # coordinates, valence and arousal.
            total_samples += targets.numel()

    # Both metrics are averaged per target coordinate over the full dataset.
    return (
        total_squared_error / total_samples,
        total_absolute_error / total_samples,
    )


def main():
    # Use a GPU when available; otherwise run the baseline on the CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # The CNN and baseline use the same fixed train/validation partition.
    train_set, validation_set = load_datasets("mock_dataset.pt")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    validation_loader = DataLoader(
        validation_set,
        batch_size=32,
        shuffle=False,
    )

    # Compute the baseline prediction from training labels only, so validation
    # targets do not leak into the model's constant prediction.
    train_targets = torch.cat(
        [targets for _, targets in train_loader],
        dim=0,
    )
    mean_targets = train_targets.mean(dim=0).to(device)

    baseline = MeanBaseline(mean_targets).to(device)
    loss_fn = nn.MSELoss()

    mse, mae = evaluate(
        baseline,
        validation_loader,
        loss_fn,
        device,
    )

    print(f"Device: {device}")
    print(f"Baseline average valence: {mean_targets[0].item():.4f}")
    print(f"Baseline average arousal: {mean_targets[1].item():.4f}")
    print(f"Baseline validation MSE: {mse:.4f}")
    print(f"Baseline validation MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
