import torch
import torch.nn.functional as F
import math


class KVPagePool:
    def __init__(self, num_pages, page_size, num_heads, head_dim, device, dtype):
        self.page_size = page_size

        self.k = torch.empty(
            num_pages, page_size, num_heads, head_dim,
            device=device, dtype=dtype
        )
        self.v = torch.empty_like(self.k)

        self.free_pages = list(range(num_pages))

    def alloc_page(self):
        assert len(self.free_pages) > 0, "KV page pool exhausted"
        return self.free_pages.pop()

    def free_page(self, page_id):
        self.free_pages.append(page_id)



class PageTable:
    def __init__(self):
        self.pages = []   # logical order of pages
        self.length = 0   # number of tokens written




def write_kv(pool, page_table, k, v):
    token_idx = page_table.length
    page_idx = token_idx // pool.page_size
    offset = token_idx % pool.page_size

    if offset == 0:
        page_id = pool.alloc_page()
        page_table.pages.append(page_id)
    else:
        page_id = page_table.pages[page_idx]

    pool.k[page_id, offset] = k
    pool.v[page_id, offset] = v

    page_table.length += 1




def paged_attention(q, pool, page_table):
    """
    q: [H, D]
    """
    scores = []
    values = []

    for page_id in page_table.pages:
        K = pool.k[page_id]    # [PAGE_SIZE, H, D]
        V = pool.v[page_id]

        s = torch.einsum("hd,thd->th", q, K) / math.sqrt(q.size(-1))
        scores.append(s)
        values.append(V)

    scores = torch.cat(scores, dim=0)   # [T, H]
    values = torch.cat(values, dim=0)   # [T, H, D]

    attn = F.softmax(scores, dim=0)
    out = torch.einsum("th,thd->hd", attn, values)
    return out


class PagedGPTInferenceEngine:
    def __init__(
        self,
        model,
        num_pages,
        page_size,
        device="cuda",
        dtype=torch.float16
    ):
        self.model = model.transformer
        self.lm_head = model.lm_head

        self.device = device
        self.dtype = dtype

        self.num_layers = model.config.n_layer
        self.num_heads = model.config.n_head
        self.d_model = model.config.n_embd
        self.head_dim = self.d_model // self.num_heads

        self.page_pool = KVPagePool(
            num_pages=num_pages,
            page_size=page_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype
        )

        self.page_tables = [PageTable() for _ in range(self.num_layers)]
        self.last_token = None


    @torch.no_grad()
    def prefill(self, input_ids):
        T = input_ids.size(1)
        device = input_ids.device

        pos = torch.arange(T, device=device)
        x = self.model.wte(input_ids) + self.model.wpe(pos)[None, :, :]

        for l, block in enumerate(self.model.h):
            pt = self.page_tables[l]
            pt.pages.clear()
            pt.length = 0

            x_ln = block.ln_1(x)
            qkv = block.attn.c_attn(x_ln)
            q, k, v = qkv.split(self.d_model, dim=-1)

            q = q.view(1, T, self.num_heads, self.head_dim)
            k = k.view(1, T, self.num_heads, self.head_dim)
            v = v.view(1, T, self.num_heads, self.head_dim)

            for t in range(T):
                write_kv(self.page_pool, pt, k[0, t], v[0, t])

            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=True
            )

            y = y.transpose(1, 2).reshape(1, T, self.d_model)
            x = x + block.attn.c_proj(y)
            x = x + block.mlp(block.ln_2(x))

        x = self.model.ln_f(x)
        logits = self.lm_head(x[:, -1])
        self.last_token = torch.argmax(logits, dim=-1)



    @torch.no_grad()
    def decode(self, max_new_tokens=32):
        generated = []

        for _ in range(max_new_tokens):
            pos = torch.tensor([self.page_tables[0].length], device=self.device)
            hidden = self.model.wte(self.last_token) + self.model.wpe(pos)

            for l, block in enumerate(self.model.h):
                pt = self.page_tables[l]

                x = block.ln_1(hidden)
                qkv = block.attn.c_attn(x)
                q, k, v = qkv.split(self.d_model, dim=-1)

                q = q.view(self.num_heads, self.head_dim)
                k = k.view(self.num_heads, self.head_dim)
                v = v.view(self.num_heads, self.head_dim)

                write_kv(self.page_pool, pt, k, v)
                y = paged_attention(q, self.page_pool, pt)

                y = y.view(1, self.d_model)
                hidden = hidden + block.attn.c_proj(y)
                hidden = hidden + block.mlp(block.ln_2(hidden))

            hidden = self.model.ln_f(hidden)
            logits = self.lm_head(hidden)
            next_token = torch.argmax(logits, dim=-1)

            generated.append(next_token.item())
            self.last_token = next_token

        return generated


