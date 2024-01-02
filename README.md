# Mixtral offloading

This project implements efficient inference of [Mixtral-8x7B models](https://mistral.ai/news/mixtral-of-experts/).

## How does it work?

In summary, we achieve efficient inference of Mixtral-8x7B models through a combination of techniques:

* **Mixed quantization with HQQ**. We apply separate quantization schemes for attention layers and experts to fit the model into the combined GPU and CPU memory.
* **MoE offloading strategy**. Each expert per layer is offloaded separately and only brought pack to GPU when needed. We store active experts in a LRU cache to reduce GPU-RAM communication when computing activations for adjacent tokens.

For more detailed information about our methods and results, please refer to our [tech-report](https://arxiv.org/abs/2312.17238).

## Running
This project offers an online demo as well as a CLI interface for local usage.

### Online
To try this demo, please use the demo notebook: [./notebooks/demo.ipynb](./notebooks/demo.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dvmazur/mixtral-offloading/blob/master/notebooks/demo.ipynb)

### CLI
To run this demo via CLI checkout this git repository, install all dependencies into a virtual env
```bash
# Prepare Venv
➜  python3 -m venv venv && source venv/bin/activate
# Install dependencies
(venv) ➜  pip3 install -r requirements.txt
```
and run the **cli.py**:
```bash
(venv) ➜  python3 cli.py
```

The CLI offers parameters to tweak the output as following
```bash
Usage: cli.py [OPTIONS]

  Generate responses using Mixtral model.

Options:
  --model-name TEXT               [default:
                                  mistralai/Mixtral-8x7B-Instruct-v0.1]
  --quantized-model-name TEXT     [default: lavawolfiee/Mixtral-8x7B-Instruct-
                                  v0.1-offloading-demo]
  --offload-per-layer INTEGER     [default: 5]
  --temperature FLOAT             [default: 0.9]
  --top-p FLOAT                   [default: 0.9]
  --max-new-tokens INTEGER        [default: 512]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.
```

## Work in progress

Some techniques described in our technical report are not yet available in this repo. However, we are actively working on adding support for them in the near future.

Some of the upcoming features are:
* Support for other quantization methods
* Speculative expert prefetching
