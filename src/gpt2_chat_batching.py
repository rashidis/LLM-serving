import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)