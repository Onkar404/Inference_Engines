import tiktoken
import torch
from model import GPT
from inference_engine import Inference_Engine

model=GPT(768,1,128,50257,12).to('cuda')
model.load_state_dict(torch.load("model-1k.pt",weights_only=True))
model.eval()

tokenizer=tiktoken.get_encoding('gpt2')

engine=Inference_Engine(model,tokenizer)

prompt='i love cats  '

generated_text=engine.generate(prompt)

print(generated_text)


