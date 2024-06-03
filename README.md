
# Unveiling-Linguistic-Regions-in-LLMs


<img src="imgs/introduction.png" alt="Introduction Image" style="width: 60%;">


## :fire: News
<!---
-->
* **[2024-05.28]** Accepted by ACL 2024. The preprint of our paper can be found [here](https://arxiv.org/abs/2402.14700).

## 💻Code Start
<h3 id="0">🛠️ Environment Configuration</h3>

```sh
conda create -n test python=3.10 -y
conda activate test

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

conda install datasets accelerate safetensors chardet cchardet wandb nvitop -c huggingface -c conda-forge -y

pip3 install transformers sentencepiece einops ninja deepspeed-kernels tqdm

pip list | grep ninja && pip3 install flash-attn

DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_CUTLASS_OPS=0 DS_BUILD_RAGGED_DEVICE_OPS=0 DS_BUILD_EVOFORMER_ATTN=0 pip install deepspeed
```

<h3 id="1">📖 Data Preprocess</h3>
<p>
  <h4> Thanks to Open-Sourced Code: </h4>
  <a href="https://github.com/zjunlp/KnowLM/tree/main/pretrain">
    <img src="https://github.com/zjunlp/KnowLM/blob/main/assets/KnowLM.png?raw=true" width="80" height="24" style="vertical-align:middle;">
  </a>
  <a href="https://github.com/zjunlp/KnowLM/tree/main/pretrain" style="vertical-align:middle; margin-left: 10px;">KnowLM-Pretrain</a>
</p>

```sh
cd data_preprocess
mkdir -p path_to_save
# preprocess your data
python preprocess-llama.py \
    --mode "write" \
    --file_path "example.jsonl" \
    --save_prefix "train" \
    --save_path "path_to_save/" \
    --language "chinese" \
    --do_keep_newlines \
    --seq_length 512 \
    --tokenizer_path 'LLaMA-2-Tokenizer' \
    --num_workers 16

# read your processed data
python preprocess-llama.py \
    --mode="read" \
    --read_path_prefix="./path_to_save/train" \
    --tokenizer_path 'LLaMA-2-Tokenizer'
```




## 🎯Generation Case
### Outlier Dimension Perturbation
> Here we use ***“Fudan University is located in”*** as prompt.
<img src="imgs/core-linguistic-output.png" alt="Core-linguistic-output Image" style="width: 65%;">
<!-- ![](imgs/core-linguistic-output.png) -->

### Monolingual Regions Removal
> Here we use ***"There are 365 days in a year and 12"*** as prompt.
<img src="imgs/monolingual-output.png" alt="Monolingual-output Image" style="width: 80%;">
<!-- ![](imgs/monolingual-output.png) -->

## 👓Regions Visualization
### Core Linguistic Region
> The **'Top 5%'** region on Attention.o and MLP.down.
<p align="center">
  <img src="imgs/core_linguistic_vertical.gif" alt="Core Linguistic" 
  style="width: 65%; ">
</p>


### Monolingual Regions
> The **'Arabic'** and **'Vietnamese'** regions on Attention.q.
<p align="center">
  <img src="imgs/monolingual_vertical.gif" alt="Monolingual" 
  style="width: 60%; ">
</p>


## 👋Others

### Reference
If you found our paper helpful, please consider citing:
```bibtex
@article{zhang2024unveiling,
  title={Unveiling Linguistic Regions in Large Language Models},
  author={Zhang, Zhihao and Zhao, Jun and Zhang, Qi and Gui, Tao and Huang, Xuanjing},
  journal={arXiv e-prints},
  pages={arXiv--2402},
  year={2024}
}
```

### Acknowledgements

Thanks to previous open-sourced repo: 
* [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)
* [KnowLM](https://github.com/zjunlp/KnowLM)
