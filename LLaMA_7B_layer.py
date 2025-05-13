import torch
import time
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import os

# -- Logits Processor 임포트 또는 fallback 정의 --
try:
    from transformers.generation_logits_process import LogitsProcessor, LogitsProcessorList
except ImportError:
    try:
        from transformers.generation_utils import LogitsProcessor, LogitsProcessorList
    except ImportError:
        # fallback: 간단한 기본 클래스를 정의합니다.
        class LogitsProcessor:
            def __call__(self, input_ids, scores):
                return scores

        class LogitsProcessorList(list):
            def __call__(self, input_ids, scores):
                for processor in self:
                    scores = processor(input_ids, scores)
                return scores

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --- Model and Tokenizer Setup ---
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Chat-tuned model variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

# (Optional) 설정 조정: 예를 들어, RoPE 설정 강제
config = LlamaConfig.from_pretrained(model_name)
if config.rope_scaling is None:
    config.rope_scaling = {"type": "linear", "factor": 1.0}
config.rope_theta = 10000.0
config.use_cache = False

# FP16을 사용하여 메모리를 절약합니다.
model = LlamaForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# --- Custom Logits Processor ---
# FP16 연산 시 수치 불안정성 문제를 완화하기 위해, 생성 시 logits를 FP32로 캐스팅합니다.
class FP32CastLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores = scores.float()
        scores = torch.where(torch.isnan(scores), torch.full_like(scores, -1e9), scores)
        scores = torch.where(torch.isinf(scores), torch.full_like(scores, -1e9), scores)
        return scores.float()

logits_processor = LogitsProcessorList([FP32CastLogitsProcessor()])

# --- Dataset Setup ---
dataset = load_dataset("Skylion007/openwebtext", split="train")
texts = [entry["text"] for entry in dataset.select(range(256))]
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
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

# --- Layer Timing Hooks Setup ---
layer_latencies = {}

def pre_hook(module, input):
    if torch.cuda.is_available():
        module.start_event = torch.cuda.Event(enable_timing=True)
        module.start_event.record()
    else:
        module.start_time = time.time()

def post_hook(module, input, output):
    if torch.cuda.is_available():
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        elapsed = module.start_event.elapsed_time(end_event)
    else:
        elapsed = (time.time() - module.start_time) * 1000
    if hasattr(module, "layer_id"):
        if module.layer_id in layer_latencies:
            layer_latencies[module.layer_id].append(elapsed)
        else:
            layer_latencies[module.layer_id] = [elapsed]

# transformer 레이어에 hook 등록 (model.model.layers 내부)
for i, layer in enumerate(model.model.layers):
    layer.layer_id = i
    layer.register_forward_pre_hook(pre_hook)
    layer.register_forward_hook(post_hook)

# --- Inference and Latency Measurement ---
num_batches_list = [4, 8, 16, 32, 64]

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
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=32,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    use_cache=False,
                    logits_processor=logits_processor
                )
                end_event.record()
                torch.cuda.synchronize()
                batch_latency = start_event.elapsed_time(end_event)
            else:
                start_time = time.time()
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=32,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    use_cache=False,
                    logits_processor=logits_processor
                )
                end_time = time.time()
                batch_latency = (end_time - start_time) * 1000
            
            total_latency += batch_latency
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print(f"Total Inference Latency for {num_batches} batches: {total_latency:.3f} ms")
    print(f"Example Output: {decoded_outputs[0][:100]}...\n")

    print("Layer-wise Total Latency (ms):")
    for layer_id in sorted(layer_latencies.keys()):
        total_latency_layer = sum(layer_latencies[layer_id])
        print(f"  Layer {layer_id}: {total_latency_layer:.3f} ms")
    print("\nFinished layer-by-layer total latency measurement for", num_batches, "batches!\n")
