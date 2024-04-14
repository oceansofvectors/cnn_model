from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import torch


def get_mean_std(loader):
    var = 0.0
    mean = 0.0
    total_images_count = 0
    pbar = tqdm(loader, desc="Calculating mean and std")

    for images, _ in pbar:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        var += ((images - images.mean(2).unsqueeze(1)) ** 2).sum([0, 2])
        total_images_count += batch_samples

    mean /= total_images_count
    std = torch.sqrt(var / total_images_count)
    return mean, std


def prepare_dataloaders(data_path, transform, batch_size, shuffle, num_workers):
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
