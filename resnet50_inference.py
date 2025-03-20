import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import time
import os
import glob
from torch.utils.data import DataLoader, Dataset

# Load the pre-trained ResNet-50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Running on: {device}")

# Load and preprocess image
image_path = "example.jpg"  # Replace with your image
img = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img = transform(img).unsqueeze(0).to(device)

# Measure latency using CUDA events
if device.type == "cuda":
    torch.cuda.synchronize()  # Ensure previous operations are done
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        output = model(img)
    end_event.record()

    torch.cuda.synchronize()  # Wait for GPU computation to finish
    latency = start_event.elapsed_time(end_event)  # Time in milliseconds
else:
    start_time = time.time()
    with torch.no_grad():
        output = model(img)
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
# Perform inference
# start_time = time.time()
# with torch.no_grad():
#     output = model(img)
# end_time = time.time()

# latency = (end_time - start_time) * 1000
print(f"Inference Latency: {latency:.3f} ms")

# Get predicted class
_, predicted_class = torch.max(output, 1)

# Load ImageNet class labels
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Print results
print(f"Predicted Class Index: {predicted_class.item()}")
print(f"Predicted Label: {labels[predicted_class.item()]}")
print("Model is on:", next(model.parameters()).device)
