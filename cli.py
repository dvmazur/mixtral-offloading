import sys

sys.path.append("mixtral-offloading")
import torch
import typer
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from torch.nn import functional as F
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer, TextStreamer
from transformers.utils import logging as hf_logging

from src.build_model import OffloadConfig, QuantConfig, build_model

app = typer.Typer()

def build_mixtral_model(quantized_model_name: str, offload_per_layer: int):
    """
    Prepare mixtral model to be used.
    """
    config = AutoConfig.from_pretrained(quantized_model_name)
    state_path = snapshot_download(quantized_model_name)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise Exception("No GPU")
    
    num_experts = config.num_local_experts
    
    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * offload_per_layer,
        buffer_size=4,
        offload_per_layer=offload_per_layer,
    )
    
    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256
    
    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )
    quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)
    
    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )
    return model

@app.command()
def generate_responses(
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    quantized_model_name: str = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo",
    offload_per_layer: int = 5, # 5 for 12GB VRAM, smaller for more VRAM
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
):
    """
    Generate responses using Mixtral model.
    """
    model = build_mixtral_model(quantized_model_name, offload_per_layer)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    past_key_values = None
    sequence = None

    seq_len = 0
    while True:
        print("User: ", end="")
        user_input = input()
        print("\n")

        user_entry = dict(role="user", content=user_input)
        input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

        if past_key_values is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
            attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

        print("Mixtral: ", end="")
        result = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            streamer=streamer,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        print("\n")

        sequence = result["sequences"]
        past_key_values = result["past_key_values"]

if __name__ == "__main__":
    app()
