
# Unveiling-Linguistic-Regions-in-LLMs


<img src="imgs/introduction.png" alt="Introduction Image" style="width: 60%;">


## :fire: News
<!---
-->
* **[2024-05.28]** Accepted by ACL 2024. The preprint of our paper can be found [here](https://arxiv.org/abs/2402.14700).

## The code will be released soon.


## Generation Case
### Outlier Dimension Perturbation
> Here we use ***“Fudan University is located in”*** as prompt.
<img src="imgs/core-linguistic-output.png" alt="Core-linguistic-output Image" style="width: 80%;">
<!-- ![](imgs/core-linguistic-output.png) -->

### Monolingual Regions Removal
> Here we use ***"There are 365 days in a year and 12"*** as prompt.
<img src="imgs/monolingual-output.png" alt="Monolingual-output Image" style="width: 80%;">
<!-- ![](imgs/monolingual-output.png) -->

## Regions Visualization
### Core Linguisitc Region
> The **'Top 5%'** region on Attention.o and MLP.down.
<p align="center">
  <img src="imgs/core_linguistic_vertical.gif" alt="Core Linguistic" 
  style="width: 75%; ">
</p>


### Monolingual Regions
> The **'Arabic'** and **'Vietnamese'** regions on Attention.q.
<p align="center">
  <img src="imgs/monolingual_vertical.gif" alt="Monolingual" 
  style="width: 70%; ">
</p>

## Reference
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

## Acknowledgements

Thanks to previous open-sourced repo: 
* [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)
* [KnowLM](https://github.com/zjunlp/KnowLM)
