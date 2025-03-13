# Cockatiel: Ensembling Synthetic and Human Preferenced Training for Detailed Video Caption
[Arxiv](https://arxiv.org/abs/2503.09279) / [Model](https://huggingface.co/Fr0zencr4nE/Cockatiel-13B) / [Dataset](https://huggingface.co/datasets/Fr0zencr4nE/Cockatiel-4K) / [Project](https://sais-fuxi.github.io/projects/cockatiel/)

## Introduction

Video Detailed Captioning (VDC) is a crucial task for vision-language bridging, enabling fine-grained descriptions of complex video content. In this paper, we first comprehensively benchmark current state-of-the-art approaches and systematically identified two critical limitations: biased capability towards specific captioning aspect and misalignment with human preferences. To address these deficiencies, we propose Cockatiel, a novel three-stage training pipeline that ensembles synthetic and human-aligned training for improving VDC performance. In the first stage, we derive a scorer from a meticulously annotated dataset to select synthetic captions high-performing on certain fine-grained video-caption alignment and human-preferred while disregarding others. Then, we train Cockatiel-13B, using this curated dataset to infuse it with assembled model strengths and human preferences. Finally, we further distill Cockatiel-8B from Cockatiel-13B for the ease of usage. Extensive quantitative and qualitative experiments reflect the effectiveness of our method, as we not only set new state-of-the-art performance on VDCSCORE in a dimension-balanced way but also surpass leading alternatives on human preference by a large margin as depicted by the human evaluation results.

## News
- &#x2705; [2025.03.13] We released the paper, model, inference code and project page of Cockatiel.

- &#x1F525; The annotated dataset and other code are still uploading, which should be finished in days.


## Checkpoints
We release [Cockatiel-13B](https://huggingface.co/Fr0zencr4nE/Cockatiel-13B) and [Cockatiel-8B](https://huggingface.co/Fr0zencr4nE/Cockatiel-8B), two video detailed captioning models and [Cockatiel-Scorer](https://huggingface.co/Fr0zencr4nE/Cockatiel-Scorer), a human-aligned quality scoring model on detailed video caption.

## Installation

Use the following scripts can perfectly reproduce the environment utilized by Cockatiel, as we have validated this.

```
sh ./environment_setup.sh 
```


## Citations

```
@misc{qin2025cockatielensemblingsynthetichuman,
      title={Cockatiel: Ensembling Synthetic and Human Preferenced Training for Detailed Video Caption}, 
      author={Luozheng Qin and Zhiyu Tan and Mengping Yang and Xiaomeng Yang and Hao Li},
      year={2025},
      eprint={2503.09279},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.09279}, 
}
```

# Acknowledgement

- [VILA](https://github.com/NVlabs/VILA) and [LLaVA](https://github.com/haotian-liu/LLaVA): These are the codebase we built upon. Besides, our README is built employing VILA's as template. Thanks for their wonderful work.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [AuroraCap](https://github.com/rese1f/aurora), these repos provide support for a cheap but holistic machine evaluation benchmark, VDCSCORE for this project.
