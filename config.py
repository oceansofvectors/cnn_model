import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
TRAIN_DATA_PATH = "data/train"
TEST_DATA_PATH = "data/test"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
NUM_WORKERS = 4

# Transformations
temp_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

final_transform = lambda mean, std: transforms.Compose([
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])
