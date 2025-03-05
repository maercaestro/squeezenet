import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from squeezenet import SqueezeNet  # Import modular SqueezeNet

########################################
# 1. Load Model
########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SqueezeNet(num_classes=10, input_channels=3, input_size=32).to(device)
model.load_state_dict(torch.load("squeezenet_cifar10.pth"))
model.eval()
print("Model loaded successfully!")

########################################
# 2. Define Preprocessing Function
########################################

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

########################################
# 3. Run Inference
########################################

def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(dim=1).item()
    return predicted_class

########################################
# 4. Example Usage
########################################

if __name__ == "__main__":
    image_path = "example_image.png"  # Replace with actual image path
    predicted_class = predict(image_path)
    print(f"Predicted Class: {predicted_class}")
