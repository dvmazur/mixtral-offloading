import typing as tp

import torch
from torch import nn

from .utils import nested_flatten, nested_pack


class MixtralExpertWrapper(nn.Module):
    def __init__(
        self,
        expert_module: tp.Any,
        device: torch.device,
    ):
        super().__init__()
        
        expert_module, self.storage = self.replace_layer_storage(expert_module, device)
        self.expert_module = lambda *args, **kwargs: expert_module(*args, **kwargs)
        
        self._register_state_dict_hook(self._add_storage_to_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._load_storage_from_state_dict_hook)
        
    @staticmethod
    def _add_storage_to_state_dict_hook(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'storage'] = torch.as_tensor(self.storage, dtype=torch.uint8)
        return state_dict
    
    def _load_storage_from_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.storage.copy_(state_dict[prefix + 'storage'].storage().untyped())
        del state_dict[prefix + 'storage']
    
    def forward(self, *args, **kwargs):
        return self.expert_module(*args, **kwargs)
    
    
    @staticmethod
    def replace_layer_storage(
        layer: tp.Any,
        device: torch.device,
    ):
        state_dict = {
            f"w{i}": {
                "W_q": getattr(layer, f"w{i}").W_q,
                "meta": getattr(layer, f"w{i}").meta,
                "bias": getattr(layer, f"w{i}").bias,
            }
            for i in range(1, 4)
        }

        storage_size = 0
        offsets = [0]

        for x in nested_flatten(state_dict):
            if not isinstance(x, torch.Tensor):
                continue
            storage_size += x.nbytes
            offsets.append(storage_size)

        storage = torch.UntypedStorage(storage_size, device=device) 

        i = 0
        new_flattened_states = list()
        for x in nested_flatten(state_dict):
            if not isinstance(x, torch.Tensor):
                new_flattened_states.append(x)
                continue

            start = offsets[i]
            end = offsets[i + 1]
            a_view = torch.as_tensor(storage[start:end], dtype=x.dtype, device=device).view(x.shape)
            a_view[...] = x
            assert a_view.data_ptr() == storage.data_ptr() + start
            i += 1
            new_flattened_states.append(a_view)

        state_dict = nested_pack(new_flattened_states, state_dict)

        for layer_id, states in state_dict.items():
            patched = getattr(layer, layer_id)
            patched.W_q = states["W_q"]
            patched.meta = states["meta"]
            patched.bias = states["bias"]
            setattr(layer, layer_id, patched)

        return layer, storage
