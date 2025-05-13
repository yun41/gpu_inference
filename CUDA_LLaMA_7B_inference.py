import torch
import time
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --- Logits Processor 임포트 또는 fallback 정의 ---
try:
    from transformers.generation_logits_process import LogitsProcessor, LogitsProcessorList
except ImportError:
    try:
        from transformers.generation_utils import LogitsProcessor, LogitsProcessorList
    except ImportError:
        # 간단한 기본 클래스 정의
        class LogitsProcessor:
            def __call__(self, input_ids, scores):
                return scores

        class LogitsProcessorList(list):
            def __call__(self, input_ids, scores):
                for processor in self:
                    scores = processor(input_ids, scores)
                return scores

# --- Model and Tokenizer Setup ---
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Chat-tuned model variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

# Load and optionally modify configuration
config = LlamaConfig.from_pretrained(model_name)
if config.rope_scaling is None:
    config.rope_scaling = {"type": "linear", "factor": 1.0}
config.rope_theta = 10000.0
# Inference에서는 캐시 사용 (속도 향상)
config.use_cache = True

# FP16을 사용하여 메모리를 절약 (Mixed precision)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# --- Custom Logits Processor ---
# FP16 연산 시 발생할 수 있는 수치 불안정성을 줄이기 위해, 생성 시 logits를 FP32로 캐스팅합니다.
class FP32CastLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores = scores.float()
        scores = torch.where(torch.isnan(scores), torch.full_like(scores, -1e9), scores)
        scores = torch.where(torch.isinf(scores), torch.full_like(scores, -1e9), scores)
        return scores.float()

logits_processor = LogitsProcessorList([FP32CastLogitsProcessor()])

# --- Dataset Setup ---
dataset = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)
texts = [entry["text"] for entry in dataset.select(range(256))]

# Ensure exactly 256 sentences (pad if needed)
while len(texts) < 256:
    texts.append("This is a placeholder sentence to maintain 256 inputs.")

print(f"Total Sentences Loaded: {len(texts)}")

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        prompt = f"User: {text}\nAssistant:"
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

# --- Inference and Latency Measurement ---
num_batches_list = [4, 8, 16, 32]

for num_batches in num_batches_list:
    batch_size = min(64, 256 // num_batches)
    print(f"\nRunning inference with {num_batches} batches (Batch Size: {batch_size})")

    dataset_instance = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset_instance, batch_size=batch_size, shuffle=False)
    
    total_latency = 0
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids, attention_mask = input_ids.to("cuda"), attention_mask.to("cuda")
            torch.cuda.empty_cache()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                # logits_processor를 추가하여 생성 시 FP32로 캐스팅하도록 함
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=0.7,
                    logits_processor=logits_processor
                )
                end_event.record()
                torch.cuda.synchronize()
                latency = start_event.elapsed_time(end_event)  # ms
            else:
                start_time = time.time()
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=0.7,
                    logits_processor=logits_processor
                )
                latency = (time.time() - start_time) * 1000  # ms
            
            total_latency += latency
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print(f"Total Inference Latency for {num_batches} batches: {total_latency:.3f} ms")
    print(f"Example Output: {decoded_outputs[0][:100]}...\n")

print("Finished LLaMA-7B inference using OpenWebText dataset!")
