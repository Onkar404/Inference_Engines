from model import GPT
import torch 
import math 
import tiktoken

model=GPT(768,12,128,50257,12).to('cuda')

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

import tiktoken
enc=tiktoken.get_encoding('gpt2')

class Dataloader():
  def __init__(self,data,B,T):
    self.B=B
    self.T=T
    self.tokens=enc.encode(data)
    self.N=len(self.tokens)
    self.current_state=0

  def next_batch(self):
    buffer=self.tokens[self.current_state:self.current_state+self.B*self.T+1]
    x=torch.tensor(buffer[:self.B*self.T]).view(self.B,self.T).to('cuda')
    y=torch.tensor(buffer[1:self.B*self.T+1]).view(self.B,self.T).to('cuda')
    self.current_state+=self.B*self.T+1
    if self.current_state+self.B*self.T+1>=self.N:
      self.current_state=0
    return x,y


train_data=text[:int(0.9*len(text))]
val_data=text[int(0.9*len(text)):]

dl=Dataloader(train_data,4,128)

model.train()
import time
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)


for i in range(1000):
  t0=time.time()
  optimizer.zero_grad()
  x,y=dl.next_batch()
  x=x.to('cuda')
  y=y.to('cuda')

  logits,loss=model(x,y)
  loss.backward()
  optimizer.step()
  torch.cuda.synchronize()
  t1=time.time()
  dt=(t1-t0)*1000
  tp=dl.B*dl.T/(t1-t0)
  if i%100==0:
        print("Loss -->",loss.item(),"\tTP(Tokens/sec) --> ",tp,"\tTime(ms) --> ",dt)


torch.save(model.state_dict(),"model-1k.pt")





