import torch
from hqq.core.quantize import Quantizer
from hqq.core.bitpack import BitPack

class PackedTensor(torch.Tensor):
    def __init__(self, t: torch.Tensor):
        self = t

    def __getattr__(self, *args, **kwargs):
        return super().__getattribute__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return super().__getitem__(*args, **kwargs)

# 4 bit to uint8
def pack_4bit_u8_common(W_q: torch.Tensor):
    height = W_q.size(0)
    assert height % 2 == 0
    
    W_q = W_q.to(torch.uint8)
    p = (W_q[::2, ...] << 4) | (W_q[1::2, ...])

    return PackedTensor(p.to(torch.uint8))

def unpack_4bit_u8_common(W_q: torch.Tensor):
    height = W_q.size(0)
    W_q = W_q.to(torch.uint8)
    result = torch.empty([2 * height] + list(W_q.shape[1:]),
                         dtype=torch.uint8, device=W_q.device)
    result[::2, ...] = (W_q >> 4)
    result[1::2, ...] = (W_q & 0b1111)

    return result

def unpack_4bit_u8_universal(W_q: torch.Tensor):
    if isinstance(W_q, PackedTensor):
        return unpack_4bit_u8_common(W_q)
    else:
        return BitPack.unpack_4bit_u8(W_q)

# 2 bit to uin8
def pack_2bit_u8_common(W_q: torch.Tensor):
    W_q = W_q.to(torch.uint8)
    height = W_q.size(0)
    p = (W_q[::4, ...] << 6) | (W_q[1::4, ...] << 4) | (W_q[2::4, ...] << 2) | (W_q[3::4, ...])

    return PackedTensor(p)

def unpack_2bit_u8_common(W_q: torch.Tensor):
    W_q = W_q.to(torch.uint8)
    height = W_q.size(0)
    result = torch.empty([4 * height] + list(W_q.shape[1:]),
                         dtype=torch.uint8, device=W_q.device)
    result[::4, ...] = (W_q >> 6) & 0b11
    result[1::4, ...] = (W_q >> 4) & 0b11
    result[2::4, ...] = (W_q >> 2) & 0b11
    result[3::4, ...] = W_q & 0b11

    return result

def unpack_2bit_u8_universal(W_q: torch.Tensor):
    if isinstance(W_q, PackedTensor):
        return unpack_2bit_u8_common(W_q)
    else:
        return BitPack.unpack_2bit_u8(W_q)

def patch_packing():
    Quantizer.pack['4bit_u8'] = pack_4bit_u8_common
    Quantizer.unpack['4bit_u8'] = unpack_4bit_u8_universal
    Quantizer.pack['2bit_u8'] = pack_2bit_u8_common
    Quantizer.unpack['2bit_u8'] = unpack_2bit_u8_universal

patch_packing()
