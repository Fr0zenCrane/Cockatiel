# Cockatiel: Ensembling Synthetic and Human Preferenced Training for Detailed Video Caption
[Arxiv](https://arxiv.org/abs/2503.09279) / [Model](https://huggingface.co/Fr0zencr4nE/Cockatiel-13B) / [Dataset](https://huggingface.co/datasets/Fr0zencr4nE/Cockatiel-4K) / [Project](https://sais-fuxi.github.io/projects/cockatiel/)

## Introduction

Video Detailed Captioning (VDC) is a crucial task for vision-language bridging, enabling fine-grained descriptions of complex video content. In this paper, we first comprehensively benchmark current state-of-the-art approaches and systematically identified two critical limitations: biased capability towards specific captioning aspect and misalignment with human preferences. To address these deficiencies, we propose Cockatiel, a novel three-stage training pipeline that ensembles synthetic and human-aligned training for improving VDC performance. In the first stage, we derive a scorer from a meticulously annotated dataset to select synthetic captions high-performing on certain fine-grained video-caption alignment and human-preferred while disregarding others. Then, we train Cockatiel-13B, using this curated dataset to infuse it with assembled model strengths and human preferences. Finally, we further distill Cockatiel-8B from Cockatiel-13B for the ease of usage. Extensive quantitative and qualitative experiments reflect the effectiveness of our method, as we not only set new state-of-the-art performance on VDCSCORE in a dimension-balanced way but also surpass leading alternatives on human preference by a large margin as depicted by the human evaluation results.

## News
- &#x2705; [2025.03.13] We released the paper, model, dataset, inference code and project page of Cockatiel.
- &#x1F525; [2025.04.22] The evaluation results for Cockatiel-13B and Cockatiel-8B (Distilled) are now available on the official [VDCSCORE](https://wenhaochai.com/aurora-web/) benchmark leaderboard! Special thanks to the Auroracap team!

## Checkpoints
We release [Cockatiel-13B](https://huggingface.co/Fr0zencr4nE/Cockatiel-13B) and [Cockatiel-8B](https://huggingface.co/Fr0zencr4nE/Cockatiel-8B), two video detailed captioning models and [Cockatiel-Scorer](https://huggingface.co/Fr0zencr4nE/Cockatiel-Scorer), a human-aligned quality scoring model on detailed video caption.

## Installation

Run our script `environment_setup.sh ` can perfectly reproduce the environment utilized by Cockatiel, as we have validated this.

```
cd Cockatiel
sh ./environment_setup.sh
git clone https://github.com/bfshi/scaling_on_scales.git
cd scaling_on_scales
python setup.py install
cd ..
```
## Inference
- Run our script `distributed_cockatiel_vidcap.py` to use Cockatiel to generate detailed video captions. This script supports distributed inference, as a consequence, it works fine under single-GPU, multi-GPU, and multi-node settings, you only need to modify `nnodes` and `nproc_per_node` and write additional code for multi-node communication.
```
python -m torch.distributed.launch \
    --nnodes=1 \
    --nproc_per_node=8 \
    --use_env \
    ddistributed_cockatiel_vidcap.py \
    --model-path Fr0zencr4nE/Cockatiel-13B \
    --conv-mode vicuna_v1  \
    --prompt_set detailed \
    --video-list-file ./demo_videos.txt \
    --caption-folder ./caption_results/
```
- To use this script to sample your videos, you need at leaset modify these parameters:
    - **prompt_set**: Normally, you can leave this parameter to its default value. These are the prompts borrowed from [VDCSCORE](https://arxiv.org/abs/2410.03051), which aims at generating detailed, short, object-focused, camera-focused, background-focused video captions, you can prompt Cockatiel to generate these dimension-specifc captions by modifying `prompt_set` to `detailed`, `short`, `main_object`, `camera`, `background`.
    - **video-list-file**: This is the `.txt` file that contains  the path to the videos to be captioned, each line should contains only one path to a video. Please be cautious, **the video's filename must be different**, since each caption will be saved with the video's name as their saved json filename.
    - **caption-folder**: This is the directory where we save the generated captions.
- If you terminate the captioning task by accident, just rerun this script again and it will automatically resume the task from the point where it was last stopped.
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
