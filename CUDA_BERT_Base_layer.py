import torch
import time
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# Load BERT-Base Model & Tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.backends.cudnn.benchmark = True  # Enable GPU optimization
print(f"Running on: {device}")

# Read exactly 256 sentences from the text file
file_name = "amazon_reviews.txt"
with open(file_name, "r", encoding="utf-8") as file:
    texts = [line.strip().lower() for line in file.readlines() if line.strip()]

# Ensure exactly 256 sentences (pad if needed)
while len(texts) < 256:
    texts.append("This is a placeholder sentence to maintain 256 inputs.")

print(f"Total Sentences Loaded: {len(texts)}")

# Define dataset class
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
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

# Function to Measure Layer-wise Latency
def measure_layer_latency(model, input_ids, attention_mask):
    """Measures the execution time per layer while running full inference."""
    
    layer_latencies = []
    x = model.bert.embeddings(input_ids)

    # Expand attention mask for multi-head self-attention
    expanded_mask = attention_mask[:, None, None, :].expand(-1, 12, -1, -1)

    with torch.no_grad():
        for i, layer in enumerate(model.bert.encoder.layer):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = layer(x, expanded_mask.to(dtype=torch.bool))[0]
            end_event.record()

            torch.cuda.synchronize()
            layer_latency = start_event.elapsed_time(end_event)

            layer_latencies.append((f"Layer {i}", layer_latency))

    return layer_latencies

# Define number of batches
num_batches_list = [1, 2, 4, 8, 16]

for num_batches in num_batches_list:
    batch_size = 256 // num_batches

    print(f"\nRunning inference with {num_batches} batches (Batch Size: {batch_size})")

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        total_latency = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            # Measure layer-wise latency
            layer_latencies = measure_layer_latency(model, input_ids, attention_mask)

            # Compute total latency per batch
            batch_latency = sum(lat for _, lat in layer_latencies)
            total_latency += batch_latency

        print(f"Total Inference Latency for {num_batches} batches: {total_latency:.3f} ms")

    # Print Layer-Wise Latencies
    print("\nLayer-wise Execution Latency:")
    for layer_name, latency in layer_latencies:
        print(f"{layer_name}: {latency:.3f} ms")

print("\nFinished BERT-Base inference.")

# Save layer-wise latencies to file
with open("bert_layerwise_latency.txt", "w") as f:
    for layer_name, latency in layer_latencies:
        f.write(f"{layer_name}: {latency:.3f} ms\n")

print("Layer-wise latency saved in 'bert_layerwise_latency.txt'")
