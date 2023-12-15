import typing as tp
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.activations import ACT2FN
from hivemind.utils import nested_flatten, nested_pack
import torch
from torch import nn
    
    
class MixtralExpertWrapper(nn.Module):
    def __init__(
        self,
        expert_module: tp.Any,
        device: torch.device,
    ):
        super().__init__()
        self.expert_module, self.storage = self.replace_layer_storage(expert_module, device)
        
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
            bytes_size = len(x.clone().storage().untyped())
            storage_size += bytes_size
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
