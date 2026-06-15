#!/usr/bin/env bash
#creat kitti_pkl and gt
#python -m pcdet.datasets.kitti.kitti_dataset_custom create_kitti_infos ../tools/cfgs/dataset_configs/kitti_dataset_custom.yaml


#########################For KITTI val set###############################

############## Train train
#CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nnodes 1 --nproc_per_node=1 --master_port 25511 train.py \
#   --gpu_id 0 --launch 'pytorch' --workers 4 \
#   --batch_size 1 --cfg_file cfgs/kitti_models/mpcf_codiff.yaml  --tcp_port 61000  \
#   --epoch 20 --max_ckpt_save_num 20 \
#   --fix_random_seed \
#
##test
#CUDA_VISIBLE_DEVICES='0' python test.py --gpu_id 0 --workers 4 --cfg_file cfgs/kitti_models/mpcf_codiff_test.yaml --batch_size 1 \
# --eval_all


############## Train Difftrain
#CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nnodes 1 --nproc_per_node=1 --master_port 25511 train.py \
#   --gpu_id 0 --launch 'pytorch' --workers 4 \
#   --batch_size 1 --cfg_file cfgs/kitti_models/mpcf_codiff.yaml  --tcp_port 61000  \
#   --epoch 20 --max_ckpt_save_num 20 \
#   --fix_random_seed \
#  # --pretrained_model ../output/kitti_models/mpcf_codiff/default/ckpt/checkpoint_epoch_53.pth
#
##test
#CUDA_VISIBLE_DEVICES='0' python test.py --gpu_id 0 --workers 4 --cfg_file cfgs/kitti_models/mpcf_codiff_test.yaml --batch_size 1 \
# --eval_all


############ Finetune #
#CUDA_VISIBLE_DEVICES='1' python -m torch.distributed.launch --nnodes 1 --nproc_per_node=1 --master_port 25530 train.py \
#   --gpu_id 1 --launch 'pytorch' --workers 8 \
#   --batch_size 1 --cfg_file cfgs/kitti_models/mpcf_codiff.yaml  --tcp_port 61000  \
#   --epoch 30 --max_ckpt_save_num 25 \
#   --fix_random_seed \
#   --pretrained_model ../output/kitti_models/mpcf_codiff/default/ckpt/checkpoint_epoch_53.pth
#
##test
#CUDA_VISIBLE_DEVICES='1' python test.py --gpu_id 1 --workers 4 --cfg_file cfgs/kitti_models/mpcf_codiff_test.yaml --batch_size 1 \
# --eval_all



##test one epoch in 'finetune' mode
CUDA_VISIBLE_DEVICES='1' python test.py --gpu_id 1 --workers 0 --cfg_file cfgs/kitti_models/mpcf_codiff_test.yaml --batch_size 1 \
--ckpt ../output/kitti_models/mpcf_codiff/default/ckpt/30Epoch_91.094-E14-063-28-finetune/checkpoint_epoch_14.pth \
#--save_to_file \
#--cal_params



######################################For KITTI test set############################

############### train-can #######################################
#CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nnodes 1 --nproc_per_node=1 --master_port 26660 train.py \
#   --gpu_id 0 --launch 'pytorch' --workers 4 \
#   --batch_size 1 --cfg_file cfgs/kitti_models/mpcf_codiff_can.yaml  --tcp_port 61000  \
#   --epoch 40 --max_ckpt_save_num 30 \
#   --fix_random_seed \


############### Diff-can #######################################
# CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nproc_per_node=1 --master_port 25530 train_diffv1.py \
#    --num_epochs 1000 --batch_size 1 --save_every 5 --lr 1e-4 --lr_scheduler cosine \
#    --data_dir ../data/nuS_for_coDIFF \
#    --save_dir ../output/DIff_checkpoints_physical \
###    --resume ../output/DIff_checkpoints_physical/checkpoint_epoch800.pth


############### Finetune-can ######################################
#CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nnodes 1 --nproc_per_node=1 --master_port 25530 train.py \
#   --gpu_id 0 --launch 'pytorch' --workers 4 \
#   --batch_size 1 --cfg_file cfgs/kitti_models/mpcf_codiff_can.yaml  --tcp_port 61000  \
#   --epoch 30 --max_ckpt_save_num 30 \
#   --fix_random_seed \
#   --pretrained_model ../output/kitti_models/checkpoint_epoch_52.pth














###Others

################################################MPCF###########################################
#python -m torch.distributed.launch --nnodes 1 --nproc_per_node=1 --master_port 25511 train.py \
#   --gpu_id 0 --launch 'pytorch' --workers 4 \
#   --batch_size 1 --cfg_file cfgs/kitti_models/voxel_rcnn_car_focal_multimodal.yaml  --tcp_port 61000  \
#   --epoch 60 --max_ckpt_save_num 25 \
#   --fix_random_seed \
####################################################Fcoal-Conv##################################
#CUDA_VISIBLE_DEVICES='1' python test.py --gpu_id 1 --workers 0 --cfg_file cfgs/kitti_models/voxel_rcnn_car_focal_multimodal.yaml --batch_size 1 \
#--ckpt ../output/kitti_models/F-conv/voxelrcnn_focal_multimodal_85.66.pth #--eval_all --cal_params
## --cal_params --ckpt ../output/kitti_models/voxel_rcnn_car/default/ckpt/voxel_rcnn_car_84.54.pth
