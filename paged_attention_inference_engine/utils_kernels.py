import triton
import triton.language as tl 
import torch

# import torch
# import time
# import triton
# import triton.language as tl

# torch.manual_seed(0)
# device = "cuda"

# # Small first (increase later)
# T = 4096
# H = 32
# D = 128
# PAGE_SIZE = 16
# NUM_PAGES = (T + PAGE_SIZE - 1) // PAGE_SIZE

# # Inputs
# K = torch.randn(T, H, D, device=device, dtype=torch.float16)
# V = torch.randn_like(K)

# block_table = torch.arange(T, device=device) // PAGE_SIZE

# K_pages_py = torch.zeros(H, NUM_PAGES, PAGE_SIZE, D, device=device, dtype=torch.float16)
# V_pages_py = torch.zeros_like(K_pages_py)

# K_pages_triton = torch.zeros_like(K_pages_py)
# V_pages_triton = torch.zeros_like(K_pages_py)




@triton.jit
def kv_write_kernel(
    K_ptr, V_ptr,
    Kp_ptr, Vp_ptr,
    block_ptr,
    T, H, D,
    PAGE_SIZE,
    stride_Kt, stride_Kh, stride_Kd,
    stride_Kph, stride_Kpp, stride_Kps, stride_Kpd,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_t = t < T
    mask_h = h < H
    mask_d = d < D

    page = tl.load(block_ptr + t, mask=mask_t, other=0)
    slot = t % PAGE_SIZE

    # ---- Load K, V ----
    k_ptrs = (
        K_ptr
        + t[:, None, None] * stride_Kt
        + h[None, :, None] * stride_Kh
        + d[None, None, :] * stride_Kd
    )

    v_ptrs = (
        V_ptr
        + t[:, None, None] * stride_Kt
        + h[None, :, None] * stride_Kh
        + d[None, None, :] * stride_Kd
    )

    mask = mask_t[:, None, None] & mask_h[None, :, None] & mask_d[None, None, :]

    k = tl.load(k_ptrs, mask=mask)
    v = tl.load(v_ptrs, mask=mask)

    # ---- Store to paged KV ----
    kp_ptrs = (
        Kp_ptr
        + h[None, :, None] * stride_Kph
        + page[:, None, None] * stride_Kpp
        + slot[:, None, None] * stride_Kps
        + d[None, None, :] * stride_Kpd
    )

    vp_ptrs = (
        Vp_ptr
        + h[None, :, None] * stride_Kph
        + page[:, None, None] * stride_Kpp
        + slot[:, None, None] * stride_Kps
        + d[None, None, :] * stride_Kpd
    )

    tl.store(kp_ptrs, k, mask=mask)
    tl.store(vp_ptrs, v, mask=mask)


def triton_kv_write(K, V, K_pages, V_pages,block_table):
    T, H, D = K.shape
    BLOCK_T=128
    BLOCK_H=8
    BLOCK_D=32
    PAGE_SIZE=16

    grid = (
        triton.cdiv(T, BLOCK_T),
        triton.cdiv(H, BLOCK_H),
        triton.cdiv(D, BLOCK_D),
    )

    kv_write_kernel[grid](
        K, V,
        K_pages, V_pages,
        block_table,
        T, H, D,
        PAGE_SIZE,
        K.stride(0), K.stride(1), K.stride(2),
        K_pages.stride(0), K_pages.stride(1),
        K_pages.stride(2), K_pages.stride(3),
        BLOCK_T,
        BLOCK_H,
        BLOCK_D,
    )



# torch.cuda.synchronize()
# t0 = time.time()
# triton_kv_write(K, V, K_pages_triton, V_pages_triton, block_table)
# torch.cuda.synchronize()
# t1 = time.time()

# print(f"Triton KV write time: {t1 - t0:.4f} s")



# def python_kv_write(K, V, K_pages, V_pages, block_table):
#     T, H, D = K.shape
#     for t in range(T):
#         page = block_table[t].item()
#         slot = t % PAGE_SIZE
#         for h in range(H):
#             K_pages[h, page, slot] = K[t, h]
#             V_pages[h, page, slot] = V[t, h]

# import time 
# torch.cuda.synchronize()
# t0 = time.time()
# python_kv_write(K, V, K_pages_py, V_pages_py, block_table)
# torch.cuda.synchronize()
# t1 = time.time()

# print(f"Python KV write time: {t1 - t0:.4f} s")


#Q,K,V[H,T,D]
# import triton
# import triton.language as tl


# @triton.jit
# def prefill_flash_attention(
#     Q_ptr, K_ptr, V_ptr, O_ptr,
#     stride_qt, stride_qh, stride_qd,
#     stride_kt, stride_kh, stride_kd,
#     stride_vt, stride_vh, stride_vd,
#     stride_ot, stride_oh, stride_od,
#     T, H,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
#     HEAD_DIM: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)   # query block
#     pid_h = tl.program_id(1)   # head

#     # ---- Query indices ----
#     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_d = tl.arange(0, HEAD_DIM)

#     mask_m = offs_m < T
#     mask_d = offs_d < HEAD_DIM

#     # ---- Load Q ----
#     q_ptrs = (
#         Q_ptr
#         + offs_m[:, None] * stride_qt
#         + pid_h * stride_qh
#         + offs_d[None, :] * stride_qd
#     )
#     q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
#     q = q.to(tl.float32)

#     # ---- Running softmax stats ----
#     m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
#     l_i = tl.zeros([BLOCK_M], tl.float32)
#     acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

#     scale = 1.0 / tl.sqrt(tl.float32(HEAD_DIM))

#     # ---- Loop over K/V blocks ----
#     for start_n in range(0, T, BLOCK_N):
#         offs_n = start_n + tl.arange(0, BLOCK_N)
#         mask_n = offs_n < T

#         # ---- Load K ----
#         k_ptrs = (
#             K_ptr
#             + offs_n[None, :] * stride_kt
#             + pid_h * stride_kh
#             + offs_d[:, None] * stride_kd
#         )
#         k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
#         k = k.to(tl.float32)

#         # ---- Compute QKᵀ ----
#         scores = tl.dot(q, k) * scale   # [BLOCK_M, BLOCK_N]

#         # ---- Causal mask ----
#         causal_mask = offs_m[:, None] >= offs_n[None, :]
#         scores = tl.where(causal_mask, scores, -float("inf"))

#         # ---- FlashAttention softmax update ----
#         m_ij = tl.max(scores, axis=1)
#         m_new = tl.maximum(m_i, m_ij)

#         p = tl.exp(scores - m_new[:, None])
#         l_new = l_i * tl.exp(m_i - m_new) + tl.sum(p, axis=1)

#         # ---- Load V ----
#         v_ptrs = (
#             V_ptr
#             + offs_n[None, :] * stride_vt
#             + pid_h * stride_vh
#             + offs_d[:, None] * stride_vd
#         )
#         v = tl.load(v_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
#         v = v.to(tl.float32)

#         acc = acc * (l_i * tl.exp(m_i - m_new) / l_new)[:, None] \
#               + tl.dot(p, v) / l_new[:, None]

#         m_i = m_new
#         l_i = l_new

#     # ---- Store output ----
#     out = acc.to(tl.float16)

#     o_ptrs = (
#         O_ptr
#         + offs_m[:, None] * stride_ot
#         + pid_h * stride_oh
#         + offs_d[None, :] * stride_od
#     )

#     tl.store(o_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])



# def triton_prefill_attention(
#     Q: torch.Tensor,
#     K: torch.Tensor,
#     V: torch.Tensor,
#     BLOCK_M: int = 16,
#     BLOCK_N: int = 64,
# ):
#     """
#     Q, K, V: [T, H, D] fp16 CUDA tensors
#     returns: O [T, H, D]
#     """

#     assert Q.is_cuda and K.is_cuda and V.is_cuda
#     assert Q.dtype == torch.float16
#     assert Q.shape == K.shape == V.shape

#     T, H, D = Q.shape

#     # Output
#     O = torch.empty_like(Q)

#     # Grid:
#     # program_id(0) -> query block
#     # program_id(1) -> head
#     grid = (
#         triton.cdiv(T, BLOCK_M),
#         H,
#     )

#     prefill_flash_attention[grid](
#         Q, K, V, O,

#         # strides for [T, H, D]
#         Q.stride(0), Q.stride(1), Q.stride(2),
#         K.stride(0), K.stride(1), K.stride(2),
#         V.stride(0), V.stride(1), V.stride(2),
#         O.stride(0), O.stride(1), O.stride(2),

#         T, H,

#         BLOCK_M=BLOCK_M,
#         BLOCK_N=BLOCK_N,
#         HEAD_DIM=D,
#     )

#     return O



        

