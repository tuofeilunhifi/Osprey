#!/bin/bash

##############################################################################################################################
# 多机多卡RDMA相关配置
### RDMA Config ####
# export NCCL_IB_ADAPTIVE_ROUTING=2
# export NCCL_SOCKET_IFNAME=^eth0
# export NCCL_IB_HCA=^mlx5_0
# export NCCL_IB_HCA=^mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
# export NCCL_SOCKET_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
export NCCL_IB_GID_INDEX=3
### RDMA Config ###

cd /mnt/pfs-guan-ssai/cv/cjy/codebase/Osprey/
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
# pip install ijson

cd /mnt/pfs-guan-ssai/cv/cjy/codebase/apex
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd /mnt/pfs-guan-ssai/cv/cjy/codebase/Osprey/

# master节点路由转IP地址
# 如 of-gptm4d16b2acc1infer-wenlian-master-0 转成 IP
# sleep 30
# waiting for system init
MASTER_IP=""
if [ "${RANK}" == "0" ];then
  while [[ "$MASTER_IP" == "" ]]
    do
        MASTER_IP=`ping ${MASTER_ADDR} -c 3 | sed '1{s/[^(]*(//;s/).*//;q}'`
        # MASTER_IP=127.0.0.1
        sleep 1
    done
else
  ## Convert DNS to IP for torch
  MASTER_IP=`getent hosts ${MASTER_ADDR} | awk '{print $1}'` # Ethernet
fi
###############################################################################################################################

export PYTHONPATH=`pwd`:$PYTHONPATH

# torchrun --nnodes=${WORLD_SIZE} \
#     --nproc_per_node=8 \
#     --rdzv_id=100 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=${MASTER_IP}:${MASTER_PORT} \
    # osprey/train/train_mem.py \
deepspeed --include localhost:4,5,6,7 osprey/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/pfs-guan-ssai/cv/cjy/models/models--sunshine-lwt--Osprey-7b-stage2/snapshots/d301f2e4575b5086253e1429f401cfaf3420e48c \
    --dataset_config ./osprey/configs/stage3.json \
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
