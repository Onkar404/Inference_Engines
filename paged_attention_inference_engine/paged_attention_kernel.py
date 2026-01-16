
#Multi Head paged attention

import triton 
import triton.language as tl 
import torch 

#q==>[H,d]
#k_cache,v_cache ==>  [H,num_pages ,page_size, head_dim]
# paged attention for single head 
@triton.jit 
def paged_attention_kernel(q_ptr,
            k_cache_ptr,v_cache_ptr,
            block_table_ptr,o_ptr,seq_len,
            stride_block_h,
            stride_kh,stride_vh,stride_qh,
            stride_kp,stride_ks,stride_kd,
            stride_vp,stride_vs,stride_vd,
            stride_oh,
            HEAD_DIM:tl.constexpr,
            PAGE_SIZE:tl.constexpr, 
            MAX_SEQ_LEN:tl.constexpr
            ):

            pid=tl.program_id(0)
            head=tl.program_id(1)
            offs_d=tl.arange(0,HEAD_DIM)
            q=tl.load(q_ptr+head*stride_qh+offs_d).to(tl.float32)

            mi = tl.full((), -float("inf"), dtype=tl.float32)

            li = tl.zeros((), dtype=tl.float32)

            acc=tl.zeros([HEAD_DIM],dtype=tl.float32)

            for t in tl.static_range(0,MAX_SEQ_LEN):
                valid = t < seq_len
                page_id = tl.load(block_table_ptr + head*stride_block_h + t )
                page_id = tl.where(valid, page_id, 0)
                slot = t % PAGE_SIZE

                k_ptr= k_cache_ptr + head*stride_kh + page_id*stride_kp + slot*stride_ks + offs_d*stride_kd
                v_ptr= v_cache_ptr + head*stride_vh + page_id*stride_vp + slot*stride_vs + offs_d*stride_vd

                k=tl.load(k_ptr,mask=valid,other=0.0).to(tl.float32)
                v=tl.load(v_ptr,mask=valid,other=0.0).to(tl.float32)

                scale=(1/tl.sqrt(tl.full((), HEAD_DIM, tl.float32)))
                score=(tl.sum(q*k,axis=0))*(scale)
                score=tl.where(valid,score,-float('inf'))

                mi_new =tl.maximum(mi,score)
                alpha =tl.exp(mi-mi_new)
                beta =tl.exp(score-mi_new)

                li = li*alpha+beta
                acc = acc*alpha + beta*v

                mi=mi_new

            out= acc / li
            tl.store(o_ptr + head*stride_oh + offs_d,out.to(tl.float16))


def paged_attention(q,k_pages,v_pages,block_table):
    assert q.is_cuda
    assert k_pages.is_cuda
    assert v_pages.is_cuda
    assert block_table.is_cuda
    o=torch.empty_like(q)
    T=block_table.shape[-1]
    stride_qh=q.stride(0)
    stride_block_h=block_table.stride(0)
    stride_oh=o.stride(0)
    stride_kh,stride_kp,stride_ks,stride_kd=k_pages.stride(0),k_pages.stride(1),k_pages.stride(2),k_pages.stride(3)
    stride_vh,stride_vp,stride_vs,stride_vd=v_pages.stride(0),v_pages.stride(1),v_pages.stride(2),v_pages.stride(3)
    HEAD_DIM=k_pages.shape[-1]
    PAGE_SIZE=k_pages.shape[-2]
    grid=(1,k_pages.shape[0])
    MAX_SEQ_LEN=128

    paged_attention_kernel[grid](q,
            k_pages,v_pages,
            block_table,o,T,
            stride_block_h,
            stride_kh,stride_vh,stride_qh,
            stride_kp,stride_ks,stride_kd,
            stride_vp,stride_vs,stride_vd,
            stride_oh,
            HEAD_DIM,
            PAGE_SIZE, 
            MAX_SEQ_LEN
            )

    return o



#Single head paged attention
# #q==>[d]
# #k_cache,v_cache ==>  [num_pages ,page_size, head_dim]
# # paged attention for single head 
# @triton.jit 
# def paged_attention_kernel(q_ptr,
#             k_cache_ptr,v_cache_ptr,
#             block_table_ptr,o_ptr,seq_len,
#             stride_kp,stride_ks,stride_kd,
#             stride_vp,stride_vs,stride_vd,
#             HEAD_DIM:tl.constexpr,
#             PAGE_SIZE:tl.constexpr, 
#             MAX_SEQ_LEN:tl.constexpr
#             ):

#             pid=tl.program_id(0)
#             offs_d=tl.arange(0,HEAD_DIM)
#             q=tl.load(q_ptr+offs_d).to(tl.float32)

#             mi = tl.full((), -float("inf"), dtype=tl.float32)

#             li = tl.zeros((), dtype=tl.float32)

#             acc=tl.zeros([HEAD_DIM],dtype=tl.float32)

#             for t in tl.static_range(0,MAX_SEQ_LEN):
#                 valid = t < seq_len
#                 page_id = tl.load(block_table_ptr + t )
#                 page_id = tl.where(valid, page_id, 0)
#                 slot = t % PAGE_SIZE

#                 k_ptr= k_cache_ptr + page_id*stride_kp + slot*stride_ks + offs_d*stride_kd
#                 v_ptr= v_cache_ptr + page_id*stride_vp + slot*stride_vs + offs_d*stride_vd

#                 k=tl.load(k_ptr,mask=valid,other=0.0).to(tl.float32)
#                 v=tl.load(v_ptr,mask=valid,other=0.0).to(tl.float32)

#                 scale=(1/tl.sqrt(tl.full((), HEAD_DIM, tl.float32)))
#                 score=(tl.sum(q*k,axis=0))*(scale)
#                 score=tl.where(valid,score,-float('inf'))

#                 mi_new =tl.maximum(mi,score)
#                 alpha =tl.exp(mi-mi_new)
#                 beta =tl.exp(score-mi_new)

#                 li = li*alpha+beta
#                 acc = acc*alpha + beta*v

#                 mi=mi_new

#             out= acc / li
#             tl.store(o_ptr+offs_d,out.to(tl.float16))


# def paged_attention(q,k_pages,v_pages,block_table):
#     assert q.is_cuda
#     assert k_pages.is_cuda
#     assert v_pages.is_cuda
#     assert block_table.is_cuda
#     o=torch.empty_like(q)
#     T=block_table.shape[0]
#     stride_kp,stride_ks,stride_kd=k_pages.stride(0),k_pages.stride(1),k_pages.stride(2)
#     stride_vp,stride_vs,stride_vd=v_pages.stride(0),v_pages.stride(1),v_pages.stride(2)
#     HEAD_DIM=k_pages.shape[-1]
#     PAGE_SIZE=k_pages.shape[-2]
#     grid=(1,)
#     MAX_SEQ_LEN=128

#     paged_attention_kernel[grid](q,
#             k_pages,v_pages,
#             block_table,o,T,
#             stride_kp,stride_ks,stride_kd,
#             stride_vp,stride_vs,stride_vd,
#             HEAD_DIM,
#             PAGE_SIZE,
#             MAX_SEQ_LEN
#             )

#     return o




# #Testing correctness 
# PAGE_SIZE = 4
# NUM_PAGES = 32
# HEAD_DIM = 64
# T = 10


# #test code 

# q = torch.rand((HEAD_DIM,), device='cuda', dtype=torch.float16)

# k_pages = torch.rand(
#     (NUM_PAGES, PAGE_SIZE, HEAD_DIM),
#     device='cuda',
#     dtype=torch.float16
# )

# v_pages = torch.rand(
#     (NUM_PAGES, PAGE_SIZE, HEAD_DIM),
#     device='cuda',
#     dtype=torch.float16
# )

# block_table = torch.randint(
#     0, NUM_PAGES,
#     (T,),
#     dtype=torch.int32,
#     device='cuda'
# )

# # y = paged_attention(q, k_pages, v_pages, block_table)
# # print(y)   # should be [HEAD_DIM]


# def ref_paged_attention(q, k_pages, v_pages, block_table):
#     D = q.shape[0]
#     T = block_table.shape[0]
#     PAGE_SIZE = k_pages.shape[1]

#     K = []
#     V = []
#     for t in range(T):
#         p = block_table[t].item()
#         s = t % PAGE_SIZE
#         K.append(k_pages[p, s])
#         V.append(v_pages[p, s])

#     K = torch.stack(K)  # [T, D]
#     V = torch.stack(V)  # [T, D]

#     scores = (q @ K.T) / (D ** 0.5)
#     attn = torch.softmax(scores, dim=-1)
#     return attn @ V


# y_triton = paged_attention(q, k_pages, v_pages, block_table)
# y_ref = ref_paged_attention(q, k_pages, v_pages, block_table)

# print("max error:", (y_triton - y_ref).abs().max())

