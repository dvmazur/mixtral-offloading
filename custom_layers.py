from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.activations import ACT2FN
from typing import Dict, Any
from accelerate import init_empty_weights

class MixtralBLockSparseTop2MLP_HQQ(nn.Module):
    def __init__(self, config: MixtralConfig, quant_config: Dict[str, Any]):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        
        with init_empty_weights():
            self.w1 = HQQLinear(nn.Linear(self.hidden_dim, self.ffn_dim, bias=False), quant_config)
            self.w2 = HQQLinear(nn.Linear(self.ffn_dim, self.hidden_dim, bias=False), quant_config)
            self.w3 = HQQLinear(nn.Linear(self.hidden_dim, self.ffn_dim, bias=False), quant_config)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states
