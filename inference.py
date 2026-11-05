
import torch
import torch.nn as nn

# Create the CNN model using the same architecture as during training
# learned weights will be loaded from the checkpoint
class MoodCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.regressor(self.features(inputs))

# Function used to assign a playlist based on predicted valence and arousal
def assign_playlist(valence, arousal, threshold=0.5):
    # The threshold turns each continuous score into a binary category
    if valence >= threshold and arousal >= threshold:
        return "High Valence / High Arousal"
    if valence >= threshold and arousal < threshold:
        return "High Valence / Low Arousal"
    if valence < threshold and arousal >= threshold:
        return "Low Valence / High Arousal"
    return "Low Valence / Low Arousal"


def main():
    # Keep inference on the GPU if available, otherwise use CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Initialize the model and load the trained weights
    model = MoodCNN().to(device)
    model.load_state_dict(
        torch.load(
            "best_cnn_model.pt",
            map_location=device,
            weights_only=True,
        )
    )
    # Set the model to evaluation mode to disable dropout and batch normalization updates
    model.eval()

    # Temporary randomized test input Mel-spectrogram
    spectrogram = torch.randn(1, 128, 128).to(device)

    # Inference needs no gradient graph, reducing memory use
    with torch.no_grad():
        prediction = model(spectrogram.unsqueeze(0))

    # There is one example in the batch and two output coordinates.
    valence = prediction[0, 0].item()
    arousal = prediction[0, 1].item()

    # Assign a playlist based on the predicted valence and arousal scores
    playlist = assign_playlist(valence, arousal)

    # print results
    print(f"Device: {device}")
    print(f"Predicted valence: {valence:.4f}")
    print(f"Predicted arousal: {arousal:.4f}")
    print(f"Assigned playlist: {playlist}")


if __name__ == "__main__":
    main()
