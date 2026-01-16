
import torch 
import torch.nn.functional as F
from paged_attention_kernel import paged_attention


class Inference_Engine:
    def __init__(self,model,tokenizer,device='cuda',max_seq_len_for_block_table=128):

        if hasattr(model,"_orig_mod"):
            model=model._orig_mod

        self.model=model
        self.tokenizer=tokenizer 
        self.model.eval()
        self.device=device 
        
        self.tok_emb=model.transformer.wte
        self.pos_emb=model.transformer.wpe

        self.blocks=model.transformer.blocks
        self.num_layers=len(model.transformer.blocks)

        self.ln_f=model.transformer.ln_f
        self.head=model.transformer.head

        self.num_heads=model.transformer.blocks[0].attn.H
        self.d_model=model.transformer.blocks[0].attn.d
        self.head_dim=self.d_model//self.num_heads 

        self.max_seq_len=max_seq_len_for_block_table
        self.PAGE_SIZE=16

        

        

    def init_kv_cache(self):
        NUM_PAGES = 128          # start small, increase later
        PAGE_SIZE = self.PAGE_SIZE      
        HEAD_DIM = self.head_dim
        num_heads=self.num_heads

        self.kv_cache=[]
        self.block_tables=[]
        self.seq_lens=[]


        for i in range(self.num_layers):

            layer_cache=[
                    PagedKVcachelayer(
                        num_pages=NUM_PAGES,
                        page_size=PAGE_SIZE,
                        head_dim=self.head_dim,
                        device=self.device
                    )
                    for _ in range(self.num_heads)
                ]
                    

            self.kv_cache.append(layer_cache)
            self.seq_lens.append(0)
            self.block_tables.append(torch.empty((self.max_seq_len,),dtype=torch.int32,device=self.device))


  


 
    def prefill(self, prompt_tokens):
        device = self.device
        prompt_tokens = prompt_tokens.to(device)
        T = prompt_tokens.shape[0]

     
        tok_embs = self.tok_emb(prompt_tokens.unsqueeze(0))     # [1, T, D]
        pos = torch.arange(T, device=device)
        pos_embs = self.pos_emb(pos)                             # [T, D]

        x = tok_embs + pos_embs.unsqueeze(0)                     # [1, T, D]

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)                                         # [1, T, D]
        self.last_hidden = x[:, -1, :]                            # [1, D]

   
        for t in range(T):
            for layer_id, block in enumerate(self.blocks):

                # project only (cheap)
                hidden_t = x[:, t:t+1, :] if False else None
                # better: recompute minimal projection
                tok = prompt_tokens[t].unsqueeze(0)
                h = self.tok_emb(tok) + self.pos_emb(torch.tensor(t, device=device))
                h = block.ln1(h)

                qkv = block.attn.c_attn(h)
                q, k, v = qkv.chunk(3, dim=-1)

                k = k.view(1, self.num_heads, self.head_dim)
                v = v.view(1, self.num_heads, self.head_dim)

                
                seq_t = self.seq_lens[layer_id]
                if seq_t % self.kv_cache[layer_id][0].page_size == 0:
                    page_id = self.kv_cache[layer_id][0].free_pages.pop(0)
                self.block_tables[layer_id][seq_t] = page_id


                for h_id in range(self.num_heads):
                    self.kv_cache[layer_id][h_id].append(
                        seq_t,
                        page_id,
                        k[:, h_id],
                        v[:, h_id]
                    )

                self.seq_lens[layer_id] += 1



    def decode(self, max_tokens):
        PAGE_SIZE=self.PAGE_SIZE
        generated_tokens = []
        hidden = self.last_hidden

        for _ in range(max_tokens):

            for layer_id, block in enumerate(self.blocks):

                t = self.seq_lens[layer_id]

                x = block.ln1(hidden)
                qkv = block.attn.c_attn(x)
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.view(1, self.num_heads, self.head_dim)
                k = k.view(1, self.num_heads, self.head_dim)
                v = v.view(1, self.num_heads, self.head_dim)

   
                attn_out = paged_attention(
                    q.squeeze(0),                               # [H, D]
                    torch.stack([c.k_pages for c in self.kv_cache[layer_id]]),
                    torch.stack([c.v_pages for c in self.kv_cache[layer_id]]),
                    self.block_tables[layer_id][:t]             # [t]
                )
                attn_out = attn_out.reshape(1, self.d_model)    
                


                attn_out = block.attn.c_proj(attn_out)
                hidden = hidden + attn_out
                hidden = hidden + block.ffnn(block.ln2(hidden))

    
                if t % self.PAGE_SIZE == 0:
                    page_id = self.kv_cache[layer_id][0].free_pages.pop(0)
                    self.kv_cache[layer_id][0].current_page = page_id
                else:
                    page_id = self.kv_cache[layer_id][0].current_page

                self.block_tables[layer_id][t] = page_id

                for h in range(self.num_heads):
                    self.kv_cache[layer_id][h].append(
                        t,
                        page_id,
                        k[:, h],
                        v[:, h]
                    )

                self.seq_lens[layer_id] += 1


            logits = self.head(self.ln_f(hidden)).squeeze(0)
            next_token = top_k_sample(logits, k=50)
            generated_tokens.append(next_token.item())

            hidden = self.tok_emb(next_token) + self.pos_emb(
                torch.tensor(self.seq_lens[0], device=self.device)
            )

        return generated_tokens


    def generate(self, prompt, max_tokens=50):
        token_ids = self.tokenizer.encode(prompt)
        token_ids = torch.tensor(
            token_ids,
            device=self.device,
            dtype=torch.long
        )

        self.init_kv_cache()
        self.prefill(token_ids)

        out_tokens = self.decode(max_tokens)

        return self.tokenizer.decode(out_tokens)







class PagedKVcachelayer:
    def __init__(self, num_pages, page_size, head_dim, device):
        self.page_size = page_size

        self.k_pages = torch.empty(
            (num_pages, page_size, head_dim),
            dtype=torch.float16,
            device=device
        )
        self.v_pages = torch.empty_like(self.k_pages)

        self.free_pages = list(range(num_pages))
        self.current_page = None

    def append(self, t, page_id, k, v):
        

        if t % self.page_size == 0:
            self.current_page = page_id

        slot = t % self.page_size
        self.k_pages[self.current_page, slot] = k.squeeze(0)
        self.v_pages[self.current_page, slot] = v.squeeze(0)


        
            



        
