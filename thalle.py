# Use a pipeline as a high-level helper
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

MODEL_ID: str = "KBTG-Labs/THaLLE-0.1-7B-fa"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
)

instruct = open('Data01.txt','r', encoding='utf-8').read()
device = "cuda" # the device to load the model onto
prompt = instruct
messages = [
    {"role": "system", "content": "นายคือผู้เชี่ยวชาญทางด้านการวิเคราะห์หุ้นและนายจะทำการวิเคราะห์หุ้นอย่างตรงไปตรงมาตามข้อมูลที่ให้ไป"},
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
    max_new_tokens=1024
)
end = time.time()
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
print(f'processing time = {end - start} sec')