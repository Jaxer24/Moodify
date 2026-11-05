import torch
from torch.utils.data import Dataset

# Defines a custom dataset called SpectrogramDataset
# By inheriting from Dataset, the class follows the interface PyTorch expects
class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, indices):
        # Apply the saved split indices once so each item lookup is direct
        self.spectrograms = spectrograms[indices]
        self.labels = labels[indices]

    def __len__(self):
        # Return the number of examples in this partition
        return len(self.labels)

    def __getitem__(self, index):
        # Return one spectrogram and its matching [valence, arousal] label
        return self.spectrograms[index], self.labels[index]


# Load the saved tensors and create the train and validation datasets
def load_datasets(path="mock_dataset.pt"):
    # Load onto CPU so data loading works regardless of where training runs
    # weights_only limits deserialization to tensors and other safe primitives
    data = torch.load(path, map_location="cpu", weights_only=True)

    # Both datasets reference the same tensors but select disjoint saved splits
    train_dataset = SpectrogramDataset(
        data["spectrograms"],
        data["labels"],
        data["train_indices"],
    )

    validation_dataset = SpectrogramDataset(
        data["spectrograms"],
        data["labels"],
        data["validation_indices"],
    )

    # Return the train and validation datasets for use in DataLoaders
    return train_dataset, validation_dataset
