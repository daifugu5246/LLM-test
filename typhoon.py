from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# import model
model_path = "scb10x/typhoon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.float16)

instr0 = open('instruction0.text', 'r').read()

instr1 = open('instruction1-2.text', 'r').read()
'''
instr0 = Role + Context + Instruction
instr1 = Role + One-shot
'''

prompt = [{"role": "user", "content": instr0 }]
tokenized_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)

out = model.generate(**tokenized_prompt, max_new_tokens=512)
result = tokenizer.decode(out[0])
print(result)