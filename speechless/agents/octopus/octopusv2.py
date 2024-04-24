
from transformers import AutoTokenizer, GemmaForCausalLM
import torch
import time

def inference(input_text):
    start_time = time.time()
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = input_ids["input_ids"].shape[1]
    outputs = model.generate(
        input_ids=input_ids["input_ids"], 
        max_length=1024,
        do_sample=False)
    generated_sequence = outputs[:, input_length:].tolist()
    res = tokenizer.decode(generated_sequence[0])
    end_time = time.time()
    return {"output": res, "latency": end_time - start_time}

model_id = "NexaAIDev/Octopus-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GemmaForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

input_text = "Take a selfie for me with front camera"
nexa_query = f"Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {input_text} \n\nResponse:"
start_time = time.time()
print("nexa model result:\n", inference(nexa_query))
print("latency:", time.time() - start_time," s")
