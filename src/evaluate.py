import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from squeezenet import SqueezeNet  # Import modular SqueezeNet

########################################
# 1. Load CIFAR-10 Test Dataset
########################################

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

########################################
# 2. Load Trained Model
########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SqueezeNet(num_classes=10, input_channels=3, input_size=32).to(device)
model.load_state_dict(torch.load("squeezenet_cifar10.pth"))
model.eval()
print("Model loaded successfully!")

########################################
# 3. Evaluation Function
########################################

def evaluate(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
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
# 4. Run Evaluation
########################################

test_loss, test_acc = evaluate(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}  | Test Accuracy: {test_acc:.2f}%")
