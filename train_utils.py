import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from cnn import ResNetGray
from config import device, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS, final_transform
from data_utils import prepare_dataloaders, get_mean_std


def train_model(train_loader, model, criterion, optimizer):
    mlflow.start_run()
    mlflow.pytorch.autolog()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(
                    f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                mlflow.log_metric("batch_loss", avg_loss, step=epoch * len(train_loader) + batch_idx)
                running_loss = 0.0

    mlflow.end_run()


def evaluate_model(test_loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    mlflow.log_metric("test_accuracy", accuracy)


def setup_and_train(train_path, test_path, temp_transform):
    temp_loader = prepare_dataloaders(train_path, temp_transform, 2000, False, 4)
    data_mean, data_std = get_mean_std(temp_loader)

    train_loader = prepare_dataloaders(train_path, final_transform(data_mean, data_std), BATCH_SIZE, True, NUM_WORKERS)
    test_loader = prepare_dataloaders(test_path, final_transform(data_mean, data_std), BATCH_SIZE, False, NUM_WORKERS)

    model = ResNetGray(7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(train_loader, model, criterion, optimizer)
    evaluate_model(test_loader, model)
