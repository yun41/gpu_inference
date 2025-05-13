import os
os.environ["USE_DHA_MEMORY"] = "1"

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Running on: {device}")

# Load input sentences
file_name = "amazon_reviews.txt"
with open(file_name, "r", encoding="utf-8") as file:
    texts = [line.strip().lower() for line in file if line.strip()]
while len(texts) < 256:
    texts.append("This is a placeholder sentence to maintain 256 inputs.")
texts = texts[:256]

# Sentiment label mapping
sentiment_labels = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0), self.texts[idx]

dataset = TextDataset(texts, tokenizer)

# DHA memory allocation
def allocate_dha_tensor(shape, dtype=torch.long):
    nbytes = int(torch.empty(shape, dtype=dtype).nbytes)
    host_buf = cuda.pagelocked_empty(nbytes, np.uint8, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
    tensor = torch.frombuffer(host_buf, dtype=dtype).reshape(shape)
    return tensor

# Run for each batch size
batch_sizes = [256, 128, 64, 32, 16, 8]
overall_latency = 0

for batch_size in batch_sizes:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"\nBatch Size: {batch_size} — Running DHA Inference...")
    total_latency = 0

    with torch.no_grad():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for input_ids, attention_mask, raw_texts in dataloader:
            # DHA memory
            host_ids = allocate_dha_tensor(input_ids.shape)
            host_mask = allocate_dha_tensor(attention_mask.shape)
            host_ids.copy_(input_ids)
            host_mask.copy_(attention_mask)

            # Transfer to GPU
            ids = host_ids.cuda(non_blocking=True)
            mask = host_mask.cuda(non_blocking=True)

            # Forward pass
            outputs = model(ids, attention_mask=mask)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # ✅ Print only first sentence in batch
            result = sentiment_labels.get(preds[0].item(), "Unknown")
            print(f'  "{raw_texts[0][:50]}..." → {result}')

        end_event.record()
        torch.cuda.synchronize()

        latency = start_event.elapsed_time(end_event)
        total_latency += latency
        print(f"  ➤ Total latency: {latency:.3f} ms for 256 inputs")

    overall_latency += total_latency

print(f"\nOverall Total Latency (with-DHA): {overall_latency:.3f} ms")
print("Finished with-DHA BERT Inference across all batch sizes.")
