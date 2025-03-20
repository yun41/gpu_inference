import torch
import time
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# Load 5-class sentiment BERT model
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

print(f"Total Sentences Loaded: {len(texts)}")  # Debugging

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
            max_length=256  # Ensure full sentence usage
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

# Sentiment label mapping (0-4 scale)
sentiment_labels = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

# Define number of batches `[1, 2, 4, 8, 16]`
num_batches_list = [1, 2, 4, 8, 16]
overall_latency = 0
for num_batches in num_batches_list:
    batch_size = 256 // num_batches  # Compute batch size dynamically

    print(f"\nRunning inference with {num_batches} batches (Batch Size: {batch_size})")

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        total_latency = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            # Measure latency
            if device.type == "cuda":
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids, attention_mask=attention_mask)
                end_event.record()

                torch.cuda.synchronize()
                latency = start_event.elapsed_time(end_event)  # Time in ms
            else:
                start_time = time.time()
                outputs = model(input_ids, attention_mask=attention_mask)
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms

            total_latency += latency

            # Convert logits to probabilities
            probabilities = F.softmax(outputs.logits, dim=1)

            # Get predicted sentiment class
            predicted_classes = torch.argmax(probabilities, dim=1)

            # Print first 5 predictions in batch
            for i in range(min(1,len(predicted_classes))):
                sentiment_result = sentiment_labels.get(predicted_classes[i].item(), "Unknown")
                print(f"Text: \"{texts[batch_idx * batch_size + i][:50]}...\" → Sentiment: {sentiment_result}")

        print(f"Total Inference Latency for {num_batches} batches: {total_latency:.3f} ms")
    overall_latency += total_latency

print(f"Overall Inference Latency: {overall_latency:.3f} ms")
print("Finished 5-level sentiment analysis using amazon_reviews.txt!")
