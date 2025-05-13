import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# Load BERT-Base Model & Tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.backends.cudnn.benchmark = True
print(f"Running on: {device}")

# Load input sentences
file_name = "amazon_reviews.txt"
with open(file_name, "r", encoding="utf-8") as file:
    texts = [line.strip().lower() for line in file.readlines() if line.strip()]
while len(texts) < 256:
    texts.append("This is a placeholder sentence to maintain 256 inputs.")
texts = texts[:256]
print(f"Total Sentences Loaded: {len(texts)}")

# Define sentiment label mapping
sentiment_labels = {
    0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"
}

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
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0), self.texts[idx]

# Measure layer-wise latency
def measure_layer_latency(model, input_ids, attention_mask):
    layer_latencies = []
    x = model.bert.embeddings(input_ids)
    extended_mask = attention_mask[:, None, None, :].expand(-1, 12, -1, -1)

    with torch.no_grad():
        for i, layer in enumerate(model.bert.encoder.layer):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            x = layer(x, extended_mask.to(dtype=torch.bool))[0]
            end_event.record()
            torch.cuda.synchronize()

            layer_latency = start_event.elapsed_time(end_event)
            layer_latencies.append((f"Layer {i}", layer_latency))

    return layer_latencies

# Inference loop with batch sizes
num_batches_list = [1, 2, 4, 8, 16]
overall_latency = 0

for num_batches in num_batches_list:
    batch_size = 256 // num_batches
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\n========== Inference (No-DHA) | Batch Size: {batch_size} ==========")
    total_latency = 0

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, raw_texts) in enumerate(dataloader):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            # 1. Print sentiment of first sentence in batch
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            print(f'  "{raw_texts[0][:50]}..." → {sentiment_labels[preds[0].item()]}')

            # 2. Layer-wise latency measurement
            layer_latencies = measure_layer_latency(model, input_ids, attention_mask)
            batch_latency = sum(lat for _, lat in layer_latencies)
            total_latency += batch_latency

        print(f"  ➤ Total latency: {total_latency:.3f} ms for 256 inputs")

    # Print average latency per layer
    print("  ➤ Last batch layer-wise latency:")
    for layer_name, latency in layer_latencies:
        print(f"    {layer_name}: {latency:.3f} ms")

    overall_latency += total_latency

print(f"\nOverall Total Latency (No-DHA): {overall_latency:.3f} ms")
print("Finished No-DHA BERT Layer-by-layer Inference.")

