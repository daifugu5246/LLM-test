from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# import model
model_path = "scb10x/typhoon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='./model-cache')
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir='./model-cache', device_map=device ,torch_dtype=torch.bfloat16)

instr0 = open('instruction0.txt', 'r', encoding='utf-8').read()
instr1 = open('instruction1-2.txt', 'r', encoding='utf-8').read()
instr2 = open('Data01.txt', 'r', encoding='utf-8').read()
'''
instr0 = Role + Context + Instruction
instr1 = Role + One-shot
'''

prompt = [{"role": "user", "content": instr1 }]
tokenized_prompt = tokenizer([instr0], return_tensors="pt").to(model.device)
start = time.time()
out = model.generate(**tokenized_prompt, max_new_tokens=4096)
end = time.time()
result = tokenizer.decode(out[0])
print(result)
print(f'processing time = {end - start} sec')