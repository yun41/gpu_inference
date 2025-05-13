import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# Load model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Running on: {device}")

# Load input text
file_name = "amazon_reviews.txt"
with open(file_name, "r", encoding="utf-8") as file:
    texts = [line.strip().lower() for line in file if line.strip()]
while len(texts) < 256:
    texts.append("This is a placeholder sentence to maintain 256 inputs.")
texts = texts[:256]

# Sentiment labels
sentiment_labels = {
    0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"
}

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0), self.texts[idx]

dataset = TextDataset(texts, tokenizer)

# Run for each batch size
batch_sizes = [256, 128, 64, 32, 16, 8]
overall_latency = 0.0

for batch_size in batch_sizes:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"\n========== Inference (No-DHA) | Batch Size: {batch_size} ==========")

    with torch.no_grad():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for input_ids, attention_mask, raw_texts in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Print only the first prediction in batch
            print(f'  "{raw_texts[0][:50]}..." → {sentiment_labels[preds[0].item()]}')

        end_event.record()
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event)

    print(f"  ➤ Total latency: {latency:.3f} ms for 256 inputs")
    overall_latency += latency

print(f"\nOverall Total Latency (No-DHA): {overall_latency:.3f} ms")
print("Finished No-DHA BERT Inference across all batch sizes.")
