import tiktoken
from model import GPT
import torch

model=GPT(768,12,128,50257,12).to('cuda')

model.load_state_dict(torch.load("model-1k.pt"))
model.eval()


enc =tiktoken.get_encoding('gpt2')
text='to be not to be '
tokens=torch.tensor(enc.encode(text)).unsqueeze(0)
tokens=tokens.to('cuda')
generated_tokens=model.generate(tokens,30)
generated_text=enc.decode(generated_tokens[0].tolist())

print(generated_text)