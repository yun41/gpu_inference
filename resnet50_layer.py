import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import glob
import numpy as np
from collections import defaultdict

# -------- 모델 설정 --------
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Running on: {device}")

# -------- 전처리 --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------- 이미지 로드 --------
image_paths = glob.glob("images/selected/*.jpg")
total_images = len(image_paths)
assert total_images > 0, "No images found."

# -------- 워밍업 --------
warmup_input = torch.randn(1, 3, 224, 224, device=device)
for _ in range(5):
    _ = model(warmup_input)
torch.cuda.synchronize()

# -------- Batch Size별 Layer-by-Layer Latency 측정 --------
batch_sizes = [1, 2, 4, 8, 16, 32]

for batch_size in batch_sizes:
    print(f"\n========== Running No-DHA Layer-by-Layer measurement (Batch size: {batch_size}) ==========")

    layer_latencies = defaultdict(float)
    layer_counts = defaultdict(int)
    total_latency = 0.0

    with torch.no_grad():
        # 이미지 배치 생성
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            current_batch_size = batch_end - batch_start
            if current_batch_size <= 0:
                continue

            # 일반 메모리로 batch 생성
            batch_tensor = torch.zeros((current_batch_size, 3, 224, 224))
            for i, img_idx in enumerate(range(batch_start, batch_end)):
                img = Image.open(image_paths[img_idx]).convert("RGB")
                img_tensor = transform(img)
                batch_tensor[i] = img_tensor

            input_tensor = batch_tensor.to(device, non_blocking=True)

            # layer-by-layer 측정
            x = input_tensor
            for name, layer in model.named_children():
                torch.cuda.synchronize()
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)

                start_evt.record()
                if name == "fc":
                    x = torch.flatten(x, 1)
                x = layer(x)
                end_evt.record()

                torch.cuda.synchronize()
                latency = start_evt.elapsed_time(end_evt)  # ms
                layer_latencies[name] += latency
                layer_counts[name] += 1

    # 결과 출력
    print("\n[No-DHA 기반 Layer-by-Layer 평균 Latency]")
    batch_total_latency = 0.0
    for name in model._modules:
        if layer_counts[name]:
            avg_latency = layer_latencies[name] / layer_counts[name]
            print(f"Layer: {name:<12} → Avg Latency: {avg_latency:.3f} ms")
            batch_total_latency += avg_latency

    print(f"\nTotal Summed Layer-wise Inference Time (Batch size {batch_size}): {batch_total_latency:.3f} ms")
    print("===================================================================")
