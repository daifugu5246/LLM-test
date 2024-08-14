from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Init Model
model_path="openthaigpt/openthaigpt-1.0.0-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path ,cache_dir='./model-cache', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir='./model-cache', trust_remote_code=True, torch_dtype=torch.float16)
model.to(device)

# Prompt
instr0 = open('instruction0.txt', 'r', encoding='utf-8').read()
instr1 = open('instruction1-1.txt', 'r', encoding='utf-8').read()

llama_prompt = instr0
inputs = tokenizer.encode(llama_prompt, return_tensors="pt")
inputs = inputs.to(device)

# Generate
start = time.time()
outputs = model.generate(inputs, max_length=1024, num_return_sequences=1)
end = time.time()
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
print(f'processing time = {end - start} sec')