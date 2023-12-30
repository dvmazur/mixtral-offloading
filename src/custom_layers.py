import copy
import functools
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.activations import ACT2FN
from typing import Dict, Any
from hqq.core.quantize import HQQLinear, Quantizer

import torch
from torch import nn
from torch.nn import functional as F

from .packing import pack_4bit_u8_common, pack_2bit_u8_common, unpack_4bit_u8_common, unpack_2bit_u8_common
from .triton_kernels import triton_matmul4_transpose, triton_matmul3_transpose, triton_matmul2_transpose


class HQQLinearTritonSavable(HQQLinear):
    def __init__(self, layer, quant_config, meta=None, **kwargs):
        """
        Example how to get meta:
        >>>> meta1 = HQQLinearSavable.get_hqq_meta((hidden_dim, ffn_dim), quant_config)
        >>>> meta2 = HQQLinearSavable.get_hqq_meta((ffn_dim, hidden_dim), quant_config)
        """
        
        assert quant_config['weight_quant_params']['nbits'] in [2, 3, 4]
        
        super().__init__(layer, quant_config, **kwargs)
        
        if not hasattr(self, 'meta'):
            assert meta is not None
            self.meta = copy.deepcopy(meta)
        
        self._register_state_dict_hook(self._add_to_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._load_from_state_dict_hook)
    
    def quantize(self, *args, **kwargs):
        super().quantize(*args, **kwargs)
        
        # repacking
        self.repack()
    
    def repack(self):
        if self.W_q.shape != self.meta['shape']:
            W_q = Quantizer.unpack[self.meta['packing']](self.W_q)
            sh = self.meta['shape']
            W_q = W_q.reshape((-1,) + sh[1:])
            W_q = W_q[:sh[0], ...]
            self.W_q = Quantizer.pack[self.meta['packing']](W_q)
    
    def forward(self, x):
        return self.forward_triton(x)
    
    def set_backend(self, backend):
        pass
    
    @torch.inference_mode()
    def forward_triton(self, x):
        assert self.ready, "model was not quantized"
        assert self.meta['axis'] == 0

        W_q, meta = self.W_q, self.meta

        del_keys = []
        if 'quant_scale' in meta and meta['quant_scale']:
            meta['scale'] = Quantizer.dequantize(meta['scale_q'], meta['meta_scale']); del_keys.append('scale')
        if 'quant_zero' in meta and meta['quant_zero']:
            meta['zero']  = Quantizer.dequantize(meta['zero_q'],  meta['meta_zero']);  del_keys.append('zero')

        K = meta['shape'][1]
        N = meta['shape'][0]
        
        if self.meta['nbits'] == 4:
            fn = triton_matmul4_transpose
        elif self.meta['nbits'] == 3:
            fn = functools.partial(triton_matmul3_transpose, N=N)
        elif self.meta['nbits'] == 2:
            fn = triton_matmul2_transpose
        else:
            raise RuntimeError(f"nbits == {self.meta['nbits']} isn't yet supported")
        
        output = fn(
            meta['group_size'], x,
            W_q.view(-1, K),
            meta['scale'].view(-1, K),
            meta['zero'].view(-1, K),
            bias=self.bias if hasattr(self, 'bias') else None,
        )

        #Cleanup
        for key in del_keys:
            del meta[key]

        return output

    # to support .forward_pytorch(...) - backward compatibility
    @torch.inference_mode()
    def dequantize(self):
        assert self.ready, "model was not quantized"
        W_q, meta = self.W_q, self.meta
        del_keys = []
        if(meta['quant_scale']):
            meta['scale'] = Quantizer.dequantize(meta['scale_q'], meta['meta_scale']); del_keys.append('scale')
        if(meta['quant_zero']):
            meta['zero']  = Quantizer.dequantize(meta['zero_q'],  meta['meta_zero']);  del_keys.append('zero')
        
        W_q_p = Quantizer.unpack[meta['packing']](W_q).half()
        W_q_p = W_q_p[:meta['shape'][0], ...]
        W_q_p = W_q_p.reshape((meta['group_size'], -1))
    
        if((meta['group_size'] is not None) and (meta['nbits']==3)):
            W_q_p = W_q_p[:meta['group_size']] if (meta['axis']==0) else W_q_p[:,:meta['group_size']]
        W_est = ((W_q_p - meta['zero'])*meta['scale']).reshape(meta['shape']) 
        
        #Cleanup
        del W_q_p
        for key in del_keys: del meta[key]
        return W_est
    
    @classmethod
    def get_hqq_meta(cls, linear_shape, quant_config):
        layer = HQQLinear(nn.Linear(*linear_shape, bias=False), quant_config)
        meta = layer.meta

        def _remove_tensors_recursive(d):
            keys = list(d.keys())

            for k in keys:
                if isinstance(d[k], torch.Tensor):
                    del d[k]
                elif isinstance(d[k], dict):
                    _remove_tensors_recursive(d[k])

        _remove_tensors_recursive(meta)

        return meta
        
    @staticmethod
    def _add_to_state_dict_hook(self, state_dict, prefix, local_metadata):
        tensor_paths = self._get_tensor_paths(self.meta)
        assert set(tensor_paths).issubset(
            {'scale_q', 'meta_scale.scale', 'meta_scale.zero', 'zero_q', 'meta_zero.scale', 'meta_zero.zero',
            'scale', 'zero'}
        )
        
        def _add(name, value):
            state_dict[prefix + name] = value
        
        _add('W_q', self.W_q)
        
        if self.bias is not None:
            _add('bias', self.bias)
        
        if 'meta_scale' in self.meta:
            _add('meta.scale_q', self.meta['scale_q'])
            _add('meta.meta_scale.scale', self.meta['meta_scale']['scale'])
            _add('meta.meta_scale.zero', self.meta['meta_scale']['zero'])
        else:
            _add('meta.scale', self.meta['scale'])
        
        if 'meta_zero' in self.meta:
            _add('meta.zero_q', self.meta['zero_q'])
            _add('meta.meta_zero.scale', self.meta['meta_zero']['scale'])
            _add('meta.meta_zero.zero', self.meta['meta_zero']['zero'])
        else:
            _add('meta.zero', self.meta['zero'])
        
        return state_dict
    
    def _load_from_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        tensor_paths = [k[len(prefix + 'meta.'):] for k in state_dict.keys() if k.startswith(prefix + 'meta.')]
        assert set(tensor_paths).issubset(
            {'scale_q', 'meta_scale.scale', 'meta_scale.zero', 'zero_q', 'meta_zero.scale', 'meta_zero.zero',
            'scale', 'zero'}
        )
        
        def _del(name):
            del state_dict[prefix + name]
        def _set(name):
            setattr(self, name, state_dict[prefix + name])
            _del(name)
        def _get(name):
            v = state_dict[prefix + name]
            _del(name)
            return v
        
        _set('W_q')
        if 'bias' in state_dict:
            _set('bias')
        else:
            self.bias = None
            
        if not hasattr(self, 'meta'):
            self.meta = {}
        
        if (prefix + 'meta.meta_scale.scale') in state_dict:
            self.meta['scale_q'] = _get('meta.scale_q')
            self.meta['quant_scale'] = True
            if not 'meta_scale' in self.meta:
                self.meta['meta_scale'] = {}
            self.meta['meta_scale'] |= {
                'scale': _get('meta.meta_scale.scale'),
                'zero': _get('meta.meta_scale.zero')
            }
        else:
            self.meta['scale'] = _get('meta.scale')
        if (prefix + 'meta.meta_zero.scale') in state_dict:
            self.meta['zero_q'] = _get('meta.zero_q')
            self.meta['quant_zero'] = True
            if not 'meta_zero' in self.meta:
                self.meta['meta_zero'] = {}
            self.meta['meta_zero'] |= {
                'scale': _get('meta.meta_zero.scale'),
                'zero': _get('meta.meta_zero.zero')
            }
        else:
            self.meta['zero'] = _get('meta.zero')
        self.ready = True
        
        # self.cuda()
        # self.in_gpu = self.W_q.device.type == 'cuda'
        # assert self.in_gpu
        
        self.repack()
        
    @classmethod
    def _get_tensor_paths(cls, state: Dict[str, Any], prefix=''):
        paths = []
        
        for k, v in state.items():
            if isinstance(v, dict):
                paths += cls._get_tensor_paths(v, prefix=k + '.')
            elif isinstance(v, torch.Tensor):
                paths.append(prefix + k)
        
        return paths
    
    def state_dict(self, *args, **kwargs):
        return nn.Module.state_dict(self, *args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        nn.Module.load_state_dict(self, *args, **kwargs)


class MixtralBLockSparseTop2MLP_HQQ(nn.Module):
    def __init__(self, config: MixtralConfig, quant_config: Dict[str, Any], meta1, meta2):
        super().__init__()
        
        self.w1 = HQQLinearTritonSavable(None, quant_config, meta1)
        self.w2 = HQQLinearTritonSavable(None, quant_config, meta2)
        self.w3 = HQQLinearTritonSavable(None, quant_config, meta1)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class SparseMoeWrapper(nn.Module):
    def __init__(self, config, layer_id, gate, expert_cache):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = gate
        self.experts = expert_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        active_experts = selected_experts.flatten().unique().tolist()

        # Loop over all available experts in the model and perform the computation on each expert
        for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
                *((self.layer_id, expert_idx) for expert_idx in active_experts), unordered=True):
            idx, top_x = torch.where(expert_mask[expert_idx])
            assert top_x.shape[0] > 0

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
