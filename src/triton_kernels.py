import triton
import triton.language as tl
import torch
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': N,
                       'BLOCK_SIZE_K': K, 'GROUP_SIZE_M': 1},
                      num_stages=S, num_warps=W) for N, K, S, W in 
        [
#             (32, 16, 1, 2),
            (32, 32, 4, 4),
#             (32, 32, 5, 2),
#             (32, 32, 5, 8),
#             (32, 128, 2, 4),
#             (64, 32, 2, 4),
#             (64, 32, 3, 4),
#             (64, 32, 4, 4),
#             (64, 32, 4, 8),
#             (64, 32, 5, 2),
#             (64, 32, 5, 8),
#             (64, 64, 3, 8),
#             (128, 32, 2, 8),
#             (128, 32, 3, 4),
#             (128, 32, 3, 8),
#             (128, 32, 4, 4),
#             (128, 32, 4, 8),
#             (256, 32, 3, 8),
#             (256, 32, 4, 4),
#             (256, 64, 3, 8),
        ]

    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul4_kernel_transpose(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (N//2, K) int32
    C is of shape (M, N) float16
    scales is of shape (G, K) float16
    zeros is of shape (G, K) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is N // groupsize.
    Set NO_GROUPS to groupsize == N, in which case G = 1 and the kernel is more efficient.

    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group # 
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the N axis 2 times
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + (offs_bn[None, :] // 2) * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    
    G = N // groupsize
    scales_ptrs = scales_ptr + (offs_bn[None, :] % G) * stride_scales_g   # (1, BLOCK_SIZE_N)
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] % G) * stride_zeros_g   # (1, BLOCK_SIZE_N)

    # shifter is used to extract the 4 bits of each element in the 8-bit word from B
    shifter = ((offs_bn + 1) % 2) * 4

    # If G == 1, scales and zeros are the same for all N, so we can load them once
    if NO_GROUPS:
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32

    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        if not NO_GROUPS:
            offs_k_scale = BLOCK_SIZE_K * k + offs_k
            ptr = scales_ptrs + offs_k_scale[:, None] * stride_scales_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            scales = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            ptr = zeros_ptrs + offs_k_scale[:, None] * stride_zeros_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        # Now we need to unpack b (which is 4-bit values) into 8-bit values
        b = (b >> shifter[None, :]) & 0xF  # Extract the 4-bit values
        b = b.to(tl.float16)
        b = (b - zeros) * scales # Scale and shift

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul4_transpose(groupsize: int, a: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, zeros: torch.FloatTensor, bias: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (M, K) float16
    qweight is of shape (N//2, K) int32
    scales is of shape (G, K) float16
    zeros is of shape (G, K) float16
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = N // groupsize
    
    C = A @ qweight.T
    Returns C of shape (..., N) float16
    """
    assert a.shape[-1] == (qweight.shape[1])
    assert a.is_contiguous(), "A must be contiguous"
    assert scales.shape[1] == zeros.shape[1]
    assert scales.shape[1] == qweight.shape[1]

    # Flatten a into (-1, K)
    x = a.view(-1, a.shape[-1])

    M, K = x.shape
    N = qweight.shape[0] * 2
    # This is based on the possible BLOCK_SIZE_Ks
#     assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
    # This is based on the possible BLOCK_SIZE_Ns
#     assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0 and N % 128 == 0 and N % 256 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
    # This is based on the possible BLOCK_SIZE_Ks
#     assert groupsize % 32 == 0 and groupsize % 64 == 0 and groupsize % 128 == 0, "groupsize must be a multiple of 32, 64, and 128"

    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul4_kernel_transpose[grid](
        x, qweight, c,
        scales, zeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        groupsize, groupsize == N,
    )
    
    # Reshape c
    c = c.view(a.shape[:-1] + (N,))  # (..., N)

    # Add bias
    if bias is not None:
        c = c + bias

    return c


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': N,
                       'BLOCK_SIZE_K': K, 'GROUP_SIZE_M': 1},
                      num_stages=S, num_warps=W) for N, K, S, W in 
        [
#             (32, 16, 1, 2),
            (32, 32, 4, 4),  # best
#             (32, 32, 5, 2),
#             (32, 32, 5, 8),
#             (32, 128, 2, 4),
#             (64, 32, 2, 4),
#             (64, 32, 3, 4),
#             (64, 32, 4, 4),
#             (64, 32, 4, 8),
#             (64, 32, 5, 2),
#             (64, 32, 5, 8),
#             (64, 64, 3, 8),
#             (128, 32, 2, 8),
#             (128, 32, 3, 4),
#             (128, 32, 3, 8),
#             (128, 32, 4, 4),
#             (128, 32, 4, 8),
#             (256, 32, 3, 8),
#             (256, 32, 4, 4),
#             (256, 64, 3, 8),
        ]

    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul2_kernel_transpose(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (N // 4, K) int8
    C is of shape (M, N) float16
    scales is of shape (G, K) float16
    zeros is of shape (G, K) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is N // groupsize.
    Set NO_GROUPS to groupsize == N, in which case G = 1 and the kernel is more efficient.

    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group # 
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the N axis 4 times
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + (offs_bn[None, :] // 4) * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    
    G = N // groupsize
    scales_ptrs = scales_ptr + (offs_bn[None, :] % G) * stride_scales_g   # (1, BLOCK_SIZE_N)
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] % G) * stride_zeros_g   # (1, BLOCK_SIZE_N)

    # shifter is used to extract the 2 bits of each element in the 8-bit word from B
    shifter = (3 - (offs_bn % 4)) * 2

    # If G == 1, scales and zeros are the same for all N, so we can load them once
    if NO_GROUPS:
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,)

    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        if not NO_GROUPS:
            offs_k_scale = BLOCK_SIZE_K * k + offs_k
            ptr = scales_ptrs + offs_k_scale[:, None] * stride_scales_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            scales = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            ptr = zeros_ptrs + offs_k_scale[:, None] * stride_zeros_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        # Now we need to unpack b (which is 4-bit values) into 8-bit values
        b = (b >> shifter[None, :]) & 0b11  # Extract the 2-bit values
        b = b.to(tl.float16)
        b = (b - zeros) * scales # Scale and shift

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul2_transpose(groupsize: int, a: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, zeros: torch.FloatTensor, bias: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (M, K) float16
    qweight is of shape (N // 4, K) int32
    scales is of shape (G, K) float16
    zeros is of shape (G, K) float16
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = N // groupsize
    
    C = A @ qweight.T
    Returns C of shape (..., N) float16
    """
    
    assert a.shape[-1] == (qweight.shape[1])
    assert a.is_contiguous(), "A must be contiguous"
    assert scales.shape[1] == zeros.shape[1]
    assert scales.shape[1] == qweight.shape[1]

    # Flatten a into (-1, K)
    x = a.view(-1, a.shape[-1])

    M, K = x.shape
    N = qweight.shape[0] * 4
    # This is based on the possible BLOCK_SIZE_Ks
#     assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
    # This is based on the possible BLOCK_SIZE_Ns
#     assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0 and N % 128 == 0 and N % 256 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
    # This is based on the possible BLOCK_SIZE_Ks
#     assert groupsize % 32 == 0 and groupsize % 64 == 0 and groupsize % 128 == 0, "groupsize must be a multiple of 32, 64, and 128"

    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul2_kernel_transpose[grid](
        x, qweight, c,
        scales, zeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        groupsize, groupsize == N,
    )
    
    # Reshape c
    c = c.view(a.shape[:-1] + (N,))  # (..., N)

    # Add bias
    if bias is not None:
        c = c + bias

    return c


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': N,
                       'BLOCK_SIZE_K': K, 'GROUP_SIZE_M': 1},
                      num_stages=S, num_warps=W) for N, K, S, W in 
        [
#             (32, 16, 1, 2),
#             (32, 32, 4, 4),
#             (32, 32, 5, 2),
            (32, 32, 5, 8),  # best
#             (32, 128, 2, 4),
#             (64, 32, 2, 4),
#             (64, 32, 3, 4),
#             (64, 32, 4, 4),
#             (64, 32, 4, 8),
#             (64, 32, 5, 2),
#             (64, 32, 5, 8),
#             (64, 64, 3, 8),
#             (128, 32, 2, 8),
#             (128, 32, 3, 4),
#             (128, 32, 3, 8),
#             (128, 32, 4, 4),
#             (128, 32, 4, 8),
#             (256, 32, 3, 8),
#             (256, 32, 4, 4),
#             (256, 64, 3, 8),
        ]

    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul3_kernel_transpose(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (ceil(N / 10), K) int32
    C is of shape (M, N) float16
    scales is of shape (G, K) float16
    zeros is of shape (G, K) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is N // groupsize.
    Set NO_GROUPS to groupsize == N, in which case G = 1 and the kernel is more efficient.

    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group # 
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
    
    # b_ptrs is set up such that it repeats elements along the N axis 10 times
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + (offs_bn[None, :] // 10) * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    
    G = N // groupsize
    scales_ptrs = scales_ptr + (offs_bn[None, :] % G) * stride_scales_g   # (1, BLOCK_SIZE_N)
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] % G) * stride_zeros_g   # (1, BLOCK_SIZE_N)

    # shifter is used to extract the 3 bits of each element in the 32-bit word from B
    shifter = (9 - (offs_bn % 10)) * 3

    # If G == 1, scales and zeros are the same for all N, so we can load them once
    if NO_GROUPS:
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,)

    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        if not NO_GROUPS:
            offs_k_scale = BLOCK_SIZE_K * k + offs_k
            ptr = scales_ptrs + offs_k_scale[:, None] * stride_scales_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            scales = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            ptr = zeros_ptrs + offs_k_scale[:, None] * stride_zeros_n  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        # Now we need to unpack b (which is 3-bit values into 32-bit values)
        b = (b >> shifter[None, :]) & 0b111  # Extract the 3-bit values
        b = b.to(tl.float16)
        b = (b - zeros) * scales # Scale and shift

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul3_transpose(groupsize: int, a: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, zeros: torch.FloatTensor, N: int, bias: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (M, K) float16
    qweight is of shape (ceil(N / 10), K) int32
    scales is of shape (G, K) float16
    zeros is of shape (G, K) float16
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = N // groupsize
    
    C = A @ qweight.T
    Returns C of shape (..., N) float16
    """
    
    assert a.shape[-1] == (qweight.shape[1])
    assert a.is_contiguous(), "A must be contiguous"
    assert scales.shape[1] == zeros.shape[1]
    assert scales.shape[1] == qweight.shape[1]

    # Flatten a into (-1, K)
    x = a.view(-1, a.shape[-1])

    M, K = x.shape
    assert 0 <= (qweight.shape[0] * 10 - N) < 10

    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul3_kernel_transpose[grid](
        x, qweight, c,
        scales, zeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        groupsize, groupsize == N,
    )
    
    # Reshape c
    c = c.view(a.shape[:-1] + (N,))  # (..., N)

    # Add bias
    if bias is not None:
        c = c + bias

    return c
