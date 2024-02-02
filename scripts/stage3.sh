#!/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH

deepspeed --include localhost:4 osprey/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/pfs-guan-ssai/cv/cjy/models/models--sunshine-lwt--Osprey-7b-stage2/snapshots/d301f2e4575b5086253e1429f401cfaf3420e48c \
    --dataset_config ./osprey/configs/stage3_test.json \
    --version v1 \
    --vision_tower /mnt/pfs-guan-ssai/cv/cjy/models/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/snapshots/39918dfbdf69ccd2172e6510a430e92337ee23e1/open_clip_pytorch_model.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir './exp/stage3' \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard" \
    --group_by_modality_length False
