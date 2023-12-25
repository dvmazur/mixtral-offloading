from .custom_layers import HQQLinearTritonSavable
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

from torch import nn


def patch_fct_hqq(linear_layer, quant_config):
    linear_layer.cuda()
    layer = HQQLinearTritonSavable(linear_layer, quant_config)
    return layer


def patch_linear_fct(linear_layer, quant_config):
    if(quant_config is None):
        return linear_layer.half().cuda()
    else:
        return patch_fct_hqq(linear_layer, quant_config)


def replace_attn_layers(model, config):
    attn_params = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=True)
    attn_params['scale_quant_params']['group_size'] = 256

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads

    shapes = [
        (hidden_size, num_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (num_heads * head_dim, hidden_size),
    ]
    shape_to_meta = {}

    for shape in shapes:
        meta = HQQLinearTritonSavable.get_hqq_meta(shape, attn_params)
        shape_to_meta[shape] = meta

    def patch_fct_hqq(shape, quant_config):
        meta = shape_to_meta[shape]
        layer = HQQLinearTritonSavable(None, quant_config, meta=meta)
        return layer

    for i in range(32):
        model.model.layers[i].block_sparse_moe.gate = nn.Linear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
        ).half().cuda()
        model.model.layers[i].self_attn.q_proj = patch_fct_hqq((hidden_size, num_heads * head_dim), attn_params)
        model.model.layers[i].self_attn.k_proj = patch_fct_hqq((hidden_size, num_key_value_heads * head_dim), attn_params)
        model.model.layers[i].self_attn.v_proj = patch_fct_hqq((hidden_size, num_key_value_heads * head_dim), attn_params)
        model.model.layers[i].self_attn.o_proj = patch_fct_hqq((hidden_size, num_heads * head_dim), attn_params)