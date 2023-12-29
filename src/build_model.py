import os
import json
from functools import cache
from dataclasses import dataclass

import torch
from torch import nn

from transformers import AutoConfig
from transformers.models.mixtral import MixtralForCausalLM


from safetensors.torch import load_file

from torch import nn
from tqdm.auto import trange

from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

from .expert_cache import ExpertCache
from .expert_wrapper import MixtralExpertWrapper
from .custom_layers import (
    HQQLinearTritonSavable,
    MixtralBLockSparseTop2MLP_HQQ,
    SparseMoeWrapper,
)
from .utils import with_default_dtype


def replace_attn_layers(model, config):
    attn_params = BaseQuantizeConfig(
        nbits=4, group_size=64, quant_zero=True, quant_scale=True
    )
    attn_params["scale_quant_params"]["group_size"] = 256

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads

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
        model.model.layers[i].block_sparse_moe.gate = (
            nn.Linear(
                config.hidden_size,
                config.num_local_experts,
                bias=False,
            )
            .half()
            .cuda()
        )
        model.model.layers[i].self_attn.q_proj = patch_fct_hqq(
            (hidden_size, num_heads * head_dim), attn_params
        )
        model.model.layers[i].self_attn.k_proj = patch_fct_hqq(
            (hidden_size, num_key_value_heads * head_dim), attn_params
        )
        model.model.layers[i].self_attn.v_proj = patch_fct_hqq(
            (hidden_size, num_key_value_heads * head_dim), attn_params
        )
        model.model.layers[i].self_attn.o_proj = patch_fct_hqq(
            (hidden_size, num_heads * head_dim), attn_params
        )


@cache
def get_default_ffn_quant_config(ffn_dim: int = 14336, hidden_dim: int = 4096):
    quant_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )

    meta1 = HQQLinearTritonSavable.get_hqq_meta((hidden_dim, ffn_dim), quant_config)
    meta2 = HQQLinearTritonSavable.get_hqq_meta((ffn_dim, hidden_dim), quant_config)

    return quant_config, meta1, meta2


def patch_fct_hqq(linear_layer, quant_config):
    linear_layer.cuda()
    layer = HQQLinearTritonSavable(linear_layer, quant_config)
    return layer


def make_empty_expert(config):
    quant_config, meta1, meta2 = get_default_ffn_quant_config()
    return MixtralBLockSparseTop2MLP_HQQ(
        config,
        quant_config,
        meta1,
        meta2,
    )


def make_and_load_expert_wrapper(
    config,
    states_dir,
    layer_idx,
    expert_idx,
    device,
):
    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.W_q"]

    state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
    expert = make_empty_expert(config)
    expert.load_state_dict(state_dict, strict=True)

    return MixtralExpertWrapper(expert, device)


def quantize_expert_inplace(expert, quant_config):
    expert.w1 = HQQLinear(expert.w1, quant_config, del_orig=True)
    expert.w2 = HQQLinear(expert.w2, quant_config, del_orig=True)
    expert.w3 = HQQLinear(expert.w3, quant_config, del_orig=True)
    expert = expert.cuda()
    return expert


def load_00_expert_state_dict(states_dir, device):
    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.0.block_sparse_moe.experts.0"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.W_q"]
    return load_file(os.path.join(states_dir, state_fpath), device=str(device))


@dataclass
class OffloadConfig:
    main_size: int
    offload_size: int
    buffer_size: int
    offload_per_layer: int


@dataclass
class QuantConfig:
    ...


def build_model(device: torch.device, offload_config: OffloadConfig, state_path: str):
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    state_dict_00 = load_00_expert_state_dict(state_path, device)

    def _make_module():
        config = AutoConfig.from_pretrained(model_name)
        expert = make_empty_expert(config)
        expert.load_state_dict(state_dict_00)
        return MixtralExpertWrapper(expert, device=device)

    with device, with_default_dtype(torch.float16):
        model = MixtralForCausalLM(
            AutoConfig.from_pretrained(
                model_name,
                num_local_experts=0,
                torch_dtype=torch.float16,
                device_map=device,
            ),
        )
    model_config = AutoConfig.from_pretrained(model_name)
    replace_attn_layers(model, model_config)
    state_index_path = os.path.join(state_path, "model.safetensors.index.json")
    with open(state_index_path) as f:
        weight_map = json.load(f)["weight_map"]

    trunk_state_path = os.path.join(
        state_path,
        weight_map["model.embed_tokens.weight"],
    )
    model.load_state_dict(load_file(trunk_state_path, device=str(device)), strict=True)

    expert_cache = ExpertCache(
        make_module=_make_module,
        main_size=offload_config.main_size,
        offload_size=offload_config.offload_size,
        buffer_size=offload_config.buffer_size,
    )
    for layer_idx in trange(model_config.num_hidden_layers, desc="Loading experts"):
        curr_layer = model.model.layers[layer_idx]
        curr_layer.block_sparse_moe = SparseMoeWrapper(
            model_config,
            layer_idx,
            curr_layer.block_sparse_moe.gate,
            expert_cache,
        )

        for expert_idx in range(model_config.num_local_experts):
            do_offload = expert_idx < offload_config.offload_per_layer

            expert_wrapper = make_and_load_expert_wrapper(
                config=model_config,
                states_dir=state_path,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                device=device,
            )

            expert_cache.add_expert(
                uid=(layer_idx, expert_idx),
                module=expert_wrapper,
                eviction_group=layer_idx,
                offload=do_offload,
            )

            del expert_wrapper
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    return model
