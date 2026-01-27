
import torch 
import transformers 
from transformers import GPT2LMHeadModel, GPT2Tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "gpt2"  # start with small
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(
    model_name,
    dtype=torch.float16
).to(device)

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

save_dir = "./gpt2_saved"

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)