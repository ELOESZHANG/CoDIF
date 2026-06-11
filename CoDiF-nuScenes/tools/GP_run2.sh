#!/usr/bin/env bash

# Create dataset info file, lidar and image gt database
#python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
#    --cfg_file ../tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
#    --version v1.0-trainval \
#    --with_cam \
#    --with_cam_gt \
#    #--share_memory

# # Create dataset for test
# python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
#     --cfg_file ../tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
#     --version v1.0-test \
#     --with_cam

#********************* Codiff 'train'**********************#

# bash ./scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/Codiff.yaml --sync_bn --pretrained_model ckpts/pretrained.pth --logger_iter_interval 100 \
# --start_epoch 10

#********************* Codiff_light 'train'**********************#73.38-70.47

 #bash ./scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/Codiff_light.yaml --sync_bn --pretrained_model ckpts/pretrained.pth --logger_iter_interval 100 \
 #--start_epoch 0

#11-75.23-72.92 12-95.0-72.8
#********************* Codiff 'diff'**********************# lr 1e-4

#  python -m torch.distributed.launch --nproc_per_node=4 --master_port 25530 train_diffv1.py \
#     --num_epochs 1000 --batch_size 1 --save_every 5 --lr 1e-4 --lr_scheduler cosine \
#     --save_dir ../output/DIff_checkpoints_physical  \
#     --resume ../output/DIff_checkpoints_physical/checkpoint_epoch910.pth


#********************* Codiff 'finetune'**********************#~
# bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/Codiff_finetune.yaml --sync_bn \
# --pretrained_model ../output/nuscenes_models/Codiff_test/default/ckpt/Zsave-train3G-test4G-E11-75.33-73.09/checkpoint_epoch_11.pth \
# --logger_iter_interval 50

##### diffusion, attentionb, grad=True.############## 73.49-70.70
#  bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/Codiff_finetune_light.yaml --sync_bn \
#  --pretrained_model ../output/nuscenes_models/Codiff_light/default/ckpt/save-train-73.38-70.47/checkpoint_epoch_10.pth \
#  --logger_iter_interval 50

#********************* Codiff test **********************# E9:75.41-73.28; E8:75.21-72.97
bash ./scripts/dist_test.sh 3 --cfg_file cfgs/nuscenes_models/Codiff_test.yaml \
--ckpt ../output/nuscenes_models/Codiff_finetune/default/ckpt/save-E9-75.41-73.28/checkpoint_epoch_9.pth --save_to_file
#--ckpt ../output/nuscenes_models/Codiff_diff/default/ckpt/save-unet-E20-preE11-75.23-72.92/checkpoint_epoch_20.pth
#--ckpt ../output/nuscenes_models/Codiff_finetune/default/ckpt/save-E9-75.41-73.28/checkpoint_epoch_9.pth \

#--ckpt ../output/nuscenes_models/Codiff_finetune/default/ckpt/save-E5-75.40-73.16/checkpoint_epoch_5.pth \
#--ckpt ../output/nuscenes_models/Codiff/default/ckpt/Zsave-train3G-test4G-E11-75.33-73.09/checkpoint_epoch_11.pth \
# --infer_time

## 
#bash ./scripts/dist_test.sh 1 --cfg_file cfgs/nuscenes_models/Codiff_test_light.yaml  \
#--ckpt ../output/nuscenes_models/Codiff_light/default/ckpt/save-train-73.38-70.47/checkpoint_epoch_10.pth --infer_time


#********************* Mambafusion **********************#ccccccccc
#bash scripts/dist_train.sh 3 --cfg_file cfgs/mambafusion_models/mamba_fusion.yaml --sync_bn --pretrained_model ckpts/pretrained.pth --logger_iter_interval 1000
#********************* Mambafusion test **********************#
#bash scripts/dist_test.sh 3 --cfg_file cfgs/nuscenes_models/mamba_fusion.yaml \
#--ckpt ../output/mambafusion_models/mamba_fusion/default/ckpt/save-lr-0.0018-NDS-74.06-71.53/checkpoint_epoch_10.pth
#--ckpt ../output/mambafusion_models/mamba_fusion/default/ckpt/checkpoint_epoch_10.pth


