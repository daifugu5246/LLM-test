# Use a pipeline as a high-level helper
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Use a pipeline as a high-level helper
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time


# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

MODEL_ID = "KBTG-Labs/THaLLE-0.1-7B-fa"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir='./model-cache')
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir='./model-cache',  device_map=device, torch_dtype=torch.bfloat16)

# Check if the parameters are quantized to 8-bit
""" for name, param in model.named_parameters():
    print(f"Parameter: {name}, Data type: {param.dtype}") """

instr0 = open('instruction0.txt', 'r', encoding='utf-8').read()
instr1 = open('instruction1-2.txt', 'r', encoding='utf-8').read()
instr2 = open('Data01.txt','r', encoding='utf-8').read()

prompt = instr1
messages = [
    {"role": "system", "content": "คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์หุ้นรายตัว และได้รับมอบหมายให้เขียนบทวิเคราะห์หุ้นตามข้อมูลที่ได้รับ"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

start = time.time()
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=4096
)
end = time.time()
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
print(f'processing time = {end - start} sec')
