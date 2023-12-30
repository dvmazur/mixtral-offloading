# Mixtral offloading

This project implements efficient inference of [Mixtral-8x7B models](https://mistral.ai/news/mixtral-of-experts/).
In short, this is achieved by quantizing the original models using HQQ in mixed quantization setups and implementing a MoE-specific offloading strategy.
You can find out the details in our [tech-report](https://arxiv.org/abs/2312.17238).

## Running

To try this demo, please use the demo notebook:  [./notebooks/demo.ipynb](./notebooks/demo.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dvmazur/mixtral-offloading/blob/master/notebooks/demo.ipynb)

For now, there is no command-line script available for running the model locally. However, you can quickly create one using the code from `src/build_model.py`. That being said, contibutions are welcome!
