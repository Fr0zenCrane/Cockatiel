# 系统默认环境变量，不建议修改
ENV="alibaba"
#ENV="huawei"

if [ ${ENV} == 'huawei' ]; then
    MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
    MASTER_PORT="6060"
    JOB_ID="1234"
    NNODES="$MA_NUM_HOSTS"
    NODE_RANK="$VC_TASK_INDEX"
    NGPUS_PER_NODE="$MA_NUM_GPUS"
fi 
if [ ${ENV} == 'alibaba' ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    NGPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi

export WANDB_MODE=offline

torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NGPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Fr0zencr4nE/Cockatiel-13B\
    --version v1 \
    --data_mixture sharegpt4_v_sft \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit -1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True

