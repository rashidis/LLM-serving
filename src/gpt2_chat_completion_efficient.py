import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

start = time.time()
tokens = model.generate(**tokenizer("A dog jumped over a", return_tensors="pt").to(device), use_cache=True, max_new_tokens=12)
print(tokenizer.decode(tokens[0]))
print(time.time() - start)