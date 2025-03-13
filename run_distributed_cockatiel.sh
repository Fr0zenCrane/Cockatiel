python -m torch.distributed.launch \
    --nnodes=1 \
    --nproc_per_node=1 \
    --use_env \
    ddistributed_cockatiel_vidcap.py \
    --model-path Fr0zencr4nE/Cockatiel-13B \
    --conv-mode vicuna_v1  \
    --prompt_set detailed \
    --video-list-file ./demo_videos.txt \
    --caption-folder ./caption_results/
