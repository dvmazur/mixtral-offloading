import numpy as np
import sys
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download
from tqdm.auto import trange
from src.build_model import build_model, OffloadConfig
import os

def initialize_model():
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    config = AutoConfig.from_pretrained(model_name)
    state_path = snapshot_download("lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo")
    device = torch.device("cuda:0")

    offload_config = OffloadConfig(
        main_size=config.num_local_experts * config.num_hidden_layers * 4 // 8,
        offload_size=config.num_local_experts * config.num_hidden_layers * 4 // 8,
        buffer_size=4,
        offload_per_layer=4,
    )

    model = build_model(device=device, offload_config=offload_config, state_path=state_path)
    return model, model_name, device

def get_user_input():
    return input("Enter your query (or type 'exit' to quit): ")

def generate_text(model, model_name, device, user_query):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}

    input_ids = tokenizer.apply_chat_template(
        [dict(role="user", content=user_query)],
        return_tensors='pt',
    ).to(device)

    inputs = dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    generated_tokens = []
    past_key_values = None

    with torch.no_grad():
        for _ in trange(100):
            out = model(**inputs, past_key_values=past_key_values, output_hidden_states=True)
            past_key_values = out.past_key_values

            probs = F.softmax(out.logits[0, -1] / 0.9, dim=-1)
            token_id = torch.multinomial(probs, 1)
            token = id_to_token[token_id.item()]

            if token[0] == '▁':
                generated_tokens.append(' ')
                token = token[1:]
            generated_tokens.append(token)

            inp = token_id.reshape(1, 1)
            inputs = dict(input_ids=inp)

    return ''.join(generated_tokens)

# Ensure environment variables are set
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia"
os.environ["LIBRARY_PATH"] = "/usr/local/cuda/lib64/stubs"

# Ensure the repository is cloned and added to sys.path
# Clone it manually or handle it in the script as needed
# sys.path.append("mixtral-offloading")

model, model_name, device = initialize_model()

while True:
    user_query = get_user_input()
    if user_query.lower() == 'exit':
        break
    response = generate_text(model, model_name, device, user_query)
    print("Response:", response)
