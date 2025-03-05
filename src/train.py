import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from squeezenet import SqueezeNet  # Import modular SqueezeNet

if __name__ == "__main__":
    ########################################
    # 1. Data Loading for CIFAR-10
    ########################################
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)  # Use 0 if issue persists
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    ########################################
    # 2. Model, Loss, and Optimizer
    ########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SqueezeNet(num_classes=10, input_channels=3, input_size=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    ########################################
    # 3. Training and Evaluation Functions
    ########################################
    def train_one_epoch(model, dataloader, optimizer, criterion, device):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples
        return avg_loss, accuracy

    def evaluate(model, dataloader, criterion, device):
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                total_correct += preds.eq(labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples
        return avg_loss, accuracy

    ########################################
    # 4. Training Loop
    ########################################
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}  | Test  Acc: {test_acc:.2f}%")
        print('-'*50)

    ########################################
    # 5. Save the Model
    ########################################
    torch.save(model.state_dict(), "squeezenet_cifar10.pth")
    print("Model saved successfully!")
