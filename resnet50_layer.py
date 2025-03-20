import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import time
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Load the pre-trained ResNet-50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode
print(f"Running on: {device}")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_path  # Return image and path

# Load images from folder
image_paths = glob.glob("images/selected/*.jpg")  # Replace with actual folder
total_images = len(image_paths)

if total_images == 0:
    print("No images found in the directory. Please check your path.")
    exit()

# Load ImageNet class labels
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Create DataLoader
dataset = ImageDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Process images one by one

# Perform layer-wise execution and measure latency
with torch.no_grad():
    total_latency = 0
    for batch, paths in dataloader:
        batch = batch.to(device)

        # Measure latency for each layer
        layer_latencies = []
        x = batch  # Input to model

        for name, layer in model.named_children():  # Process each layer separately
            if device.type == "cuda":
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                if name == "fc":  
                    x = torch.flatten(x, start_dim=1)  # Flatten before passing to fully connected layer
                x = layer(x)  # Pass input through current layer

                end_event.record()

                torch.cuda.synchronize()
                latency = start_event.elapsed_time(end_event)  # Time in ms
            else:
                start_time = time.time()
                x = layer(x)  # Pass input through current layer
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms

            layer_latencies.append((name, latency))

        # Print latency per layer
        # print("\nLayer-wise Execution Latency:")
        # for name, latency in layer_latencies:
        #     print(f"Layer: {name} → Latency: {latency:.3f} ms")

        total_latency += sum(lat for _, lat in layer_latencies)

    print(f"\nTotal Inference Latency: {total_latency:.3f} ms")

# Save results to a file
with open("resnet50_layer_latency.txt", "w") as f:
    for name, latency in layer_latencies:
        f.write(f"Layer: {name} → Latency: {latency:.3f} ms\n")

print("Layer-wise execution latency saved in 'resnet50_layer_latency.txt'.")

# Plot layer latency
def plot_layer_latency(layer_latencies):
    layer_names = [name for name, _ in layer_latencies]
    layer_times = [lat for _, lat in layer_latencies]

    plt.figure(figsize=(12, 6))
    plt.bar(layer_names, layer_times)
    plt.xticks(rotation=90)
    plt.ylabel("Latency (ms)")
    plt.xlabel("Layers")
    plt.title("ResNet-50 Layer-wise Execution Latency")
    plt.tight_layout()
    plt.savefig("resnet50_layer_latency.png")  # Save the plot instead of displaying it
    print("Layer-wise execution latency plot saved as 'resnet50_layer_latency.png'")

plot_layer_latency(layer_latencies)
