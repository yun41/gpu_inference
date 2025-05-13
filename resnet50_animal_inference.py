import os
os.environ["USE_DHA_MEMORY"] = "1"

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import glob
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

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

# -------- 클래스 레이블 --------
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f]

# -------- DHA 메모리 할당 --------
def allocate_host_mapped_tensor(shape, dtype=torch.float32):
    nbytes = int(torch.empty(shape, dtype=dtype).nbytes)
    host_buf = cuda.pagelocked_empty(nbytes, np.uint8, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
    tensor = torch.frombuffer(host_buf, dtype=dtype).reshape(shape)
    return tensor, host_buf

# -------- 워밍업 --------
warmup_input = torch.randn(1, 3, 224, 224, device=device)
for _ in range(5):
    _ = model(warmup_input)
torch.cuda.synchronize()

# -------- 동적 배치 기반 실험 --------
batch_counts = [32, 64, 128, 256, 512]
overall_latency = 0.0

for num_batches in batch_counts:
    batch_size = max(1, total_images // num_batches)
    batch_size = min(batch_size, total_images)
    print(f"\nRunning DHA inference with {num_batches} batches (Batch Size: {batch_size})")

    total_time = 0.0
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_images)
            current_batch_size = end_idx - start_idx
            if current_batch_size <= 0:
                continue

            # DHA 메모리 할당
            host_tensor, _ = allocate_host_mapped_tensor((current_batch_size, 3, 224, 224))

            # 이미지 로드 및 저장
            for i, img_idx in enumerate(range(start_idx, end_idx)):
                img = Image.open(image_paths[img_idx]).convert("RGB")
                img_tensor = transform(img)
                host_tensor[i] = img_tensor

            input_tensor = host_tensor.cuda(non_blocking=True)

            # 이벤트로 지연 측정
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_evt.record()
            output = model(input_tensor)
            end_evt.record()
            torch.cuda.synchronize()

            elapsed = start_evt.elapsed_time(end_evt)
            total_time += elapsed

    overall_latency += total_time
    print(f"  Total Inference Latency: {total_time:.3f} ms")

print(f"\nOverall Inference Latency: {overall_latency:.3f} ms")
print("Finished DHA-based dynamic batch inference.")
