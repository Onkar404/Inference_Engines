import torch
import torch.nn as nn 
import torch.nn.functional as F


class Inference_Engine:
    def __init__(self,model,tokenizer,device='cuda'):
        '''particular model has been taken , adpater class will be addded later'''
        if hasattr(model,'_orig_mod'):
            model=model._orig_mod

        self.model=model
        self.tokenizer=tokenizer
        self.model.eval()
        self.device=device
        


        
        self.tok_emb=model.transformer.wte
        self.pos_emb=model.transformer.wpe
        self.block_size = self.pos_emb.num_embeddings

        self.blocks=model.transformer.blocks 
        self.num_layers=len(self.blocks)

        self.ln_f=model.transformer.ln_f
        self.head=model.transformer.head

        self.kv_cache = None
        self.seq_len = None

    def init_kv_cache(self):
        '''will initilaize kv cache for all layers can be accesed as kv_cache[i] for layer i'''
        self.kv_cache=[KV_cache_Layer() for _ in range(self.num_layers)]





    def prefill(self,prompt_tokens):
        '''run all the layers steps on prompt and fill the kv cahe for paged attention '''

        device=self.device
        prompt_tokens=prompt_tokens.to(device)
        seq_len=0

        for token in prompt_tokens:
            token=token.unsqueeze(0)

            tok_emb=self.tok_emb(token)
            pos_emb=self.pos_emb(torch.tensor(seq_len, device=device))

            hidden=tok_emb + pos_emb

            for layer_id , block in enumerate(self.blocks):

                x = block.ln1(hidden)
                qkv=block.attn.c_attn(x)
                q,k,v=qkv.chunk(3,dim=-1)

                k_cache,v_cache=self.kv_cache[layer_id].get()

                if k_cache is not None:
                    attn_out=causal_attention(q,k_cache,v_cache)
                else:
                    attn_out = q

                attn_out=block.attn.c_proj(attn_out)

                hidden= hidden + attn_out

                hidden= hidden + block.ffnn(block.ln2(hidden))

                self.kv_cache[layer_id].append(k,v)

            seq_len+=1

        self.seq_len=seq_len
        self.last_hidden=hidden




        



    def decode(self,max_tokens):
        generated_tokens=[]
        device=self.device
        hidden=self.last_hidden

        for step in range(max_tokens):

            if self.seq_len >= self.block_size:
    # keep only last (block_size - 1) tokens in KV cache
                for layer in self.kv_cache:
                    layer.k = layer.k[-(self.block_size - 1):]
                    layer.v = layer.v[-(self.block_size - 1):]
                self.seq_len = self.block_size - 1


            for layer_id ,block in enumerate(self.blocks):
                x=block.ln1(hidden)

                qkv=block.attn.c_attn(x)
                q,k,v=qkv.chunk(3,dim=-1)

                k_cache,v_cache=self.kv_cache[layer_id].get()

                attn_out=causal_attention(q,k_cache,v_cache)

                attn_out=block.attn.c_proj(attn_out)

                hidden= hidden + attn_out

                hidden= hidden + block.ffnn(block.ln2(hidden))

                self.kv_cache[layer_id].append(k,v)

            temperature=1.0

            logits=self.head(self.ln_f(hidden))
            logits=logits.squeeze(0)
            # next_token=torch.argmax(logits,dim=-1)
            probs = F.softmax(logits / temperature, dim=-1)
            
            # next_token = torch.multinomial(probs, num_samples=1)
            next_token = top_k_sample(logits, k=50)

            generated_tokens.append(next_token.item())

                    # ---- prepare next step ----
            hidden = self.tok_emb(next_token)
            pos = self.pos_emb(torch.tensor(self.seq_len, device=device))
            hidden = hidden + pos

            self.seq_len += 1

        return generated_tokens








    def generate(self,prompt,max_tokens=50):
        ''' encode the prompt prefill and then decode for max_tokens
        and then decode the out tokens form tokenizer.decode and return text
        '''

        token_ids=self.tokenizer.encode(prompt)
        token_ids=torch.tensor(token_ids).to('cuda')

        self.init_kv_cache()

        self.prefill(token_ids)
        out_tokens = self.decode(max_tokens)

        text = self.tokenizer.decode(out_tokens)
        print(out_tokens)

        return text



        

        



class KV_cache_Layer:
    def __init__(self):
        self.k=[]
        self.v=[]

    def append(self,k,v):
        self.k.append(k)
        self.v.append(v)

    def get(self):
        if len(self.k)==0:
            return None,None 
        return torch.stack(self.k),torch.stack(self.v)


def causal_attention(Q, K, V):
    """
    Q: [1, D]
    K: [T, 1, D]
    V: [T, 1, D]
    """
    # squeeze batch dim
    Q = Q.squeeze(0)        # [D]
    K = K.squeeze(1)        # [T, D]
    V = V.squeeze(1)        # [T, D]

    scores = torch.matmul(Q, K.T) / (Q.shape[-1] ** 0.5)  # [T]
    attn = torch.softmax(scores, dim=-1)                  # [T]
    out = torch.matmul(attn, V)                            # [D]

    return out.unsqueeze(0)  # [1, D]

    
def top_k_sample(logits, k=50):
    values, indices = torch.topk(logits, k)
    probs = torch.softmax(values, dim=-1)
    idx = torch.multinomial(probs, 1)
    return indices[idx]
    
