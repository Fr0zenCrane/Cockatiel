# 系统默认环境变量，不建议修改
python -m torch.distributed.launch \
    --nnodes=1 \
    --nproc_per_node=2\
    --use_env \
    distributed_cockatiel_scoring.py \
    \
    --model-path /cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/qlz/VILA/checkpoints/sais_vidcap_rewardData_3_6k_bs_16/merged \
    --conv-mode vicuna_v1  \
    --num-video-frames 8 \
    --data-json-file ./demo_scoring_videos.json \
    --result-folder ./scoring_results