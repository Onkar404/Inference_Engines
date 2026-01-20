
import torch 
import torch.nn.functional as F
from paged_attention_kernel import paged_attention
from utils_kernels import triton_kv_write


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
        tokens = prompt_tokens.to(device)
        T = tokens.shape[0]

        # embeddings
        tok_emb = self.tok_emb(tokens)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = tok_emb + pos_emb                     # [T, D_model]

        self.init_kv_cache()

        for layer_id, block in enumerate(self.blocks):

            # 1. LayerNorm
            x_ln = block.ln1(x)                   # [T, D_model]

            # 2. QKV projection
            qkv = block.attn.c_attn(x_ln)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(T, self.num_heads, self.head_dim)
            k = k.view(T, self.num_heads, self.head_dim)
            v = v.view(T, self.num_heads, self.head_dim)

            for t in range(T):
                self.block_tables[layer_id][t] = t // self.PAGE_SIZE

            # 3. WRITE KV CACHE (your Triton kernel)
            K_pages = torch.stack([c.k_pages for c in self.kv_cache[layer_id]])
            V_pages = torch.stack([c.v_pages for c in self.kv_cache[layer_id]])

            triton_kv_write(
                k,
                v,
                K_pages,
                V_pages,
                self.block_tables[layer_id],
            )
            self.seq_lens[layer_id] = T

            # 4. ATTENTION (FlashAttention on projections)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1),
                k.transpose(0, 1),
                v.transpose(0, 1),
                is_causal=True,
            ).transpose(0, 1).contiguous()

            attn_out = attn_out.view(T, -1)

            # 5. Output projection + residual
            x = x + block.attn.c_proj(attn_out)

            # 6. FFN
            x = x + block.ffnn(block.ln2(x))

        self.last_hidden = x[-1].unsqueeze(0)


    def decode(self, max_tokens):
        PAGE_SIZE = self.PAGE_SIZE
        generated_tokens = []
        hidden = self.last_hidden                     # [1, D]

        for _ in range(max_tokens):

            for layer_id, block in enumerate(self.blocks):

                t = self.seq_lens[layer_id]           # current token index

                # ---- QKV projection ----
                x = block.ln1(hidden)                 # [1, D]
                qkv = block.attn.c_attn(x)
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.view(1, self.num_heads, self.head_dim)
                k = k.view(1, self.num_heads, self.head_dim)
                v = v.view(1, self.num_heads, self.head_dim)

                # ---- PAGED ATTENTION (READ ONLY) ----
                attn_out = paged_attention(
                    q.squeeze(0),                     # [H, D]
                    torch.stack([c.k_pages for c in self.kv_cache[layer_id]]),
                    torch.stack([c.v_pages for c in self.kv_cache[layer_id]]),
                    self.block_tables[layer_id][:t],  # history only
                )                                     # [H, D]

                attn_out = attn_out.view(1, self.d_model)
                attn_out = block.attn.c_proj(attn_out)

                hidden = hidden + attn_out
                hidden = hidden + block.ffnn(block.ln2(hidden))

                # ---- WRITE KV FOR NEW TOKEN (PERMANENT LOGIC) ----
                page_id = t // PAGE_SIZE
                slot = t % PAGE_SIZE

                # fill block table ONCE for this token
                self.block_tables[layer_id][t] = page_id

                # write KV
                for h in range(self.num_heads):
                    self.kv_cache[layer_id][h].k_pages[page_id, slot] = k[:, h]
                    self.kv_cache[layer_id][h].v_pages[page_id, slot] = v[:, h]

                self.seq_lens[layer_id] += 1

            # ---- sample next token ----
            logits = self.head(self.ln_f(hidden)).squeeze(0)
            next_token = torch.argmax(logits)
            generated_tokens.append(next_token.item())

            hidden = (
                self.tok_emb(next_token)
                + self.pos_emb(torch.tensor(self.seq_lens[0], device=self.device))
            )

        return generated_tokens


    def generate(self, prompt, max_tokens=50):
        token_ids = self.tokenizer.encode(prompt)
        token_ids = torch.tensor(
            token_ids,
            device=self.device,
            dtype=torch.long
        )


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
        # t: global token index
        # page_id: allocated by inference engine

        if t % self.page_size == 0:
            self.current_page = page_id

        slot = t % self.page_size
        self.k_pages[self.current_page, slot] = k.squeeze(0)
        self.v_pages[self.current_page, slot] = v.squeeze(0)

      
    
def top_k_sample(logits, k=50):
    values, indices = torch.topk(logits, k)
    probs = torch.softmax(values, dim=-1)
    idx = torch.multinomial(probs, 1)
    return indices[idx]
    



def project_qkv(x, qkv_proj, num_heads):
    """
    x: [T, D_model]
    returns Q, K, V: [T, H, D]
    """
    T, D_model = x.shape
    qkv = qkv_proj(x)              # [T, 3*D_model]
    q, k, v = qkv.chunk(3, dim=-1)

    head_dim = D_model // num_heads

    q = q.view(T, num_heads, head_dim)
    k = k.view(T, num_heads, head_dim)
    v = v.view(T, num_heads, head_dim)

    return q, k, v
