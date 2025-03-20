import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import time
import glob
from torch.utils.data import DataLoader, Dataset

# Load the pre-trained ResNet-50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
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
image_paths = glob.glob("images/selected/*.jpg")  # Replace with your actual folder
total_images = len(image_paths)

# Load ImageNet class labels
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Iterate through batch counts: 1, 2, 4, 8, 16
batch_counts = [1, 2, 4, 8, 16]
overall_latency = 0
for num_batches in batch_counts:
    batch_size = max(1, total_images // num_batches)  # Dynamically calculate batch size

    # Ensure batch size does not exceed total images
    if batch_size > total_images:
        batch_size = total_images  

    print(f"\nRunning inference with {num_batches} batches (Batch Size: {batch_size})")

    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Perform batch inference with latency measurement
    with torch.no_grad():
        total_latency = 0
        for batch, paths in dataloader:
            batch = batch.to(device)

            # Measure latency
            if device.type == "cuda":
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                output = model(batch)
                end_event.record()

                torch.cuda.synchronize()
                latency = start_event.elapsed_time(end_event)  # Time in ms
            else:
                start_time = time.time()
                output = model(batch)
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms

            total_latency += latency
            _, predicted_classes = torch.max(output, 1)

            # Print results for the first image in each batch (for reference)
            # print(f"Image: {paths[0]} → Predicted Label: {labels[predicted_classes[0].item()]}")
        overall_latency += total_latency
        print(f"Total Inference Latency: {total_latency:.3f} ms")
        
    
print(f"Overall Inference Latency: {overall_latency:.3f} ms")
print("Finished dynamic batch inference!")

