# Mixtral offloading

This project implements efficient inference of [Mixtral-8x7B models](https://mistral.ai/news/mixtral-of-experts/).

## How does it work?

In summary, we achieve efficient inference of Mixtral-8x7B models through a combination of techniques:

* **Mixed quantization with HQQ**. We apply separate quantization schemes for attention layers and experts to fit the model into the combined GPU and CPU memory.
* **MoE offloading strategy**. Each expert per layer is offloaded separately and only brought pack to GPU when needed. We store active experts in a LRU cache to reduce GPU-RAM communication when computing activations for adjacent tokens.

For more detailed information about our methods and results, please refer to our [tech-report](https://arxiv.org/abs/2312.17238).

## Running

To try this demo, please use the demo notebook: [./notebooks/demo.ipynb](./notebooks/demo.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dvmazur/mixtral-offloading/blob/master/notebooks/demo.ipynb)

For now, there is no command-line script available for running the model locally. However, you can create one using the demo notebook as a reference. That being said, contributions are welcome!

## Work in progress

Some techniques described in our technical report are not yet available in this repo. However, we are actively working on adding support for them in the near future.

Some of the upcoming features are:
* Support for other quantization methods
* Speculative expert prefetching
