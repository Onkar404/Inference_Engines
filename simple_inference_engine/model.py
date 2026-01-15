import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class causal_attn(nn.Module):
  def __init__(self,T,d,H,bias=False,dropout=0.2):
    super().__init__()
    assert d % H == 0
    self.c_attn=nn.Linear(d,3*d,bias=bias)
    self.c_proj=nn.Linear(d,d,bias=bias)
    self.c_dropout=nn.Dropout(dropout)
    self.res_dropout=nn.Dropout(dropout)
    self.H=H
    self.d=d
    self.T=T
    self.register_buffer("mask",torch.tril(torch.ones(T,T)).view(1,1,T,T))
    self.dropout=dropout

  def forward(self,x):
    B,T,_=x.shape
    q,k,v=self.c_attn(x).split(self.d,dim=2)

    q=q.view(B,T,self.H,self.d // self.H).transpose(1,2)
    k=k.view(B,T,self.H,self.d // self.H).transpose(1,2)
    v=v.view(B,T,self.H,self.d // self.H).transpose(1,2)

    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)

    # attn=(q@k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
    # attn=attn.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf'))
    # attn=F.softmax(attn,dim=-1)
    # attn=self.c_dropout(attn)
    # y=attn@v

    y=y.transpose(1,2).contiguous().view(B,T,self.d)
    y=self.res_dropout(self.c_proj(y))
    return y



class FFNN(nn.Module):
  def __init__(self,d,bias=False,dropout=0.2):
    super().__init__()
    self.c_fc=nn.Linear(d,4*d,bias=bias)
    self.gelu=nn.GELU()
    self.c_proj=nn.Linear(4*d,d,bias=bias)
    self.dropout=nn.Dropout(dropout)

  def forward(self,x):
    x=self.c_fc(x)
    x=self.gelu(x)
    x=self.c_proj(x)
    x=self.dropout(x)
    return x


        
class Block(nn.Module):
  def __init__(self,d,H,T,bias=False,dropout=0.2):
    super().__init__()

    self.ln1=nn.LayerNorm(d)
    # self.ln1=TritonLayerNorm(d)
    self.attn=causal_attn(T,d,H,bias=bias,dropout=dropout)
    self.ln2=nn.LayerNorm(d)
    # self.ln2=TritonLayerNorm(d)
    self.ffnn=FFNN(d,bias=bias,dropout=dropout)

  def forward(self,x):
    x=x+self.attn(self.ln1(x))
    x=x+self.ffnn(self.ln2(x))
    return x


class Block(nn.Module):
  def __init__(self,d,H,T,bias=False,dropout=0.2):
    super().__init__()

    self.ln1=nn.LayerNorm(d)
    # self.ln1=TritonLayerNorm(d)
    self.attn=causal_attn(T,d,H,bias=bias,dropout=dropout)
    self.ln2=nn.LayerNorm(d)
    # self.ln2=TritonLayerNorm(d)
    self.ffnn=FFNN(d,bias=bias,dropout=dropout)

  def forward(self,x):
    x=x+self.attn(self.ln1(x))
    x=x+self.ffnn(self.ln2(x))
    return x




class GPT(nn.Module):

    def __init__(self,
        d,
        H,
        T,
        V,
        layers,
        block_size=1024,
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        V: size of the token vocabulary
        layers: number of decoder-only blocks
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(V, d), # token embeddings
            wpe=nn.Embedding(T, d), # position embeddings
            drop=nn.Dropout(dropout),
            blocks=nn.ModuleList([Block(d, H, T, bias, dropout) for _ in range(layers)]),
            ln_f=nn.LayerNorm(d),
            # ln_f=TritonLayerNorm(d),
            head=nn.Linear(d, V, bias=bias),
        ))
        self.block_size = block_size
        self.transformer.wte.weight=self.transformer.head.weight
        self.apply(self.init_weights)
        


    def init_weights(self,module):
      if isinstance(module,nn.Linear):
        torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
        if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
      elif isinstance(module,nn.Embedding):
        torch.nn.init.normal_(module.weight,mean=0,std=0.2)

    def forward(self, idx, targets=None):
        # idx is a [B, T] matrix of token indices
        # targets is a [B, T] matrix of target (next) token indices
        device = idx.device
        _, T = idx.size() # [B, T]
        assert T <= self.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # generate token and position embeddings
        tok_emb = self.transformer.wte(idx) # [B, T, d]
        pos_emb = self.transformer.wpe(pos) # [T, d]
        x = self.transformer.drop(tok_emb + pos_emb)

        # pass through all decoder-only blocks
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x) # final layer norm

        if targets is not None:
            # compute the loss if we are given targets
            logits = self.transformer.head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # only look at last token if performing inference
            logits = self.transformer.head(x[:, [-1], :])
            loss = None

        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



