import torch

# Fix the random seed for reproducibility
SEED = 42

# Dataset configuration
DATASET_SIZE = 1000
TRAIN_SIZE = 800
OUTPUT_PATH = "mock_dataset.pt"


def main():
    # A local generator makes this file's random data and split reproducible
    # since running the script multiple times creates the same random values every time
    # without changing PyTorch's global random-number state
    generator = torch.Generator().manual_seed(SEED)

    # Creates a tensor containing fake spectrogram data
    # Each example is a single-channel 128x128 spectrogram-shaped tensor
    spectrograms = torch.randn(
        DATASET_SIZE,
        1,
        128,
        128,
        generator=generator,
    )

    # Creates the target labels for the 1,000 examples
    # Target Labels are uniformly sampled continuous [valence, arousal] values
    labels = torch.rand(
        DATASET_SIZE,
        2,
        generator=generator,
    )  # data order [valence, arousal]

    # Save the partition once so training and baseline runs use identical
    # examples instead of generating a new split each time

    # # Shuffle all example indices using the seeded generator to create a random 
    # train/validation split instead of simply putting the first 800 examples into training
    # Each index appears exactly once
    indices = torch.randperm(DATASET_SIZE, generator=generator)

    # Store mock data in 4 tensors: spectrograms, labels, train_indices, validation_indices
    dataset = {
        "spectrograms": spectrograms,
        "labels": labels,
        # Indicate the first TRAIN_SIZE shuffled indices are for training
        "train_indices": indices[:TRAIN_SIZE],
        # Indicate all remaining shuffled indices are for validation
        "validation_indices": indices[TRAIN_SIZE:],
    }

    # Store tensors and split indices together for the shared data loader
    torch.save(dataset, OUTPUT_PATH)

    # print confirmation and dataset sizes
    print(f"Saved {OUTPUT_PATH}")
    print(f"Training samples: {TRAIN_SIZE}")
    print(f"Validation samples: {DATASET_SIZE - TRAIN_SIZE}")


if __name__ == "__main__":
    main()
