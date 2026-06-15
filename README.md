<div align="center">
  <h1>CoDIF: Conditional Diffusion Fusion with Soft Alignment <br> for Robust Multimodal 3D Object Detection</h1>

  <p>
    <a href="mailto:202311050810@std.uestc.edu.cn">Pan Gao</a><sup>1</sup>,
    <a href="mailto:freecjm2003@scpolicec.edu.cn">Jianmei Cheng</a><sup>2†</sup>,
    <a href="mailto:chfei@uestc.edu.cn">Chun Fei</a><sup>3</sup>,
    <a href="mailto:renshuai@uestc.edu.cn">Shuai Ren</a><sup>1</sup>,
    <a href="mailto:202421050526@std.uestc.edu.cn">Yuling Fan</a><sup>1</sup>,
    <a href="mailto:pingzh@uestc.edu.cn">Ping Zhang</a><sup>1,4,5†</sup>
  </p>
  <p>
    <sup>1</sup>School of Optoelectronic Science and Engineering, University of Electronic Science and Technology of China<br/>
    <sup>2</sup>Intelligent Policing Key Laboratory of Sichuan Province, Sichuan Police College<br/>
    <sup>3</sup>Industrial Technology Research Institute, University of Electronic Science and Technology of China<br/>
    <sup>4</sup>Shenzhen Institute for Advanced Study, University of Electronic Science and Technology of China<br/>
    <sup>5</sup>Yibin Institute, University of Electronic Science and Technology of China
  </p>
  <p>
    <sup>†</sup>Corresponding author.
  </p>
</div>

## Abstract

We present CoDIF, a multimodal 3D object detection framework that replaces rigid calibration-based alignment with a **soft alignment** mechanism via conditional diffusion fusion.


## Overview

![CoDIF Overview](Figs/Figure1.jpg)

---

## 🔥 News

- **[2026.06]** Code of CoDIF is released.

---

## Installation

### NuScenes Setup

```bash
conda create -n codif python=3.8

# You can install CUDA 11.8 (if it isn't already on your system) by running:
# conda install --channel "nvidia/label/cuda-11.8.0" cuda
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install extra dependency
pip install -r requirements.txt

# Install nuscenes-devkit
pip install nuscenes-devkit==1.0.5 torchinfo

# Develop
python setup.py develop

cd selective_scan
python setup.py develop

cd mamba_diffv/mamba
python setup.py develop

python -m pip install causal-conv1d==1.2.0.post2
```

### KITTI Setup

```bash
conda create -n codif_kitti python=3.8
conda activate codif_kitti

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install spconv-cu113
pip install -r requirements.txt
python -m pip install causal-conv1d==1.2.0.post2

python setup.py develop

cd pcdet/ops/iou3d/cuda_op
python setup.py develop

cd pcdet/ops/mamba
python setup.py install

cd Thop_FLOPs/pytorch-OpCounter-master
pip install .

pip install matplotlib open3d
pip install flash-attn --no-build-isolation
pip install xformers==0.0.20
```

---

## Dataset Preparation

### NuScenes

Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and organize the downloaded files as follows:

```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval
├── pcdet
├── tools
```

Generate the data infos by running the following command:

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval \
    --with_cam \
    --with_cam_gt \

```

The format of the generated data is as follows:

```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval
│   │   │   │── img_gt_database_10sweeps_withvelo
│   │   │   │── gt_database_10sweeps_withvelo
│   │   │   │── nuscenes_10sweeps_withvelo_lidar.npy (optional)
│   │   │   │── nuscenes_10sweeps_withvelo_img.npy (optional)
│   │   │   │── nuscenes_infos_10sweeps_train.pkl
│   │   │   │── nuscenes_infos_10sweeps_val.pkl
│   │   │   │── nuscenes_dbinfos_10sweeps_withvelo.pkl
├── pcdet
├── tools
```

### KITTI

Please prepare the KITTI dataset as described in [Voxel-R-CNN](https://github.com/djiajunustc/Voxel-R-CNN) or [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Then replace the corresponding folders and files with those we provide in [Google Drive](https://drive.google.com/drive/folders/1nrgj1pAYGfNSb3MPLrkuLW27WWyJc68a?usp=sharing) / [Baidu Netdisk](https://pan.baidu.com/s/1uq-xD6e5mGUdYm7ROvV6Jw?pwd=swre) (unzip before replacing).

The `depth_dense_twise` folder contains dense depth maps, and the `depth_pseudo_rgbseguv_twise` folder contains pseudo point clouds. Generate pseudo point clouds from dense depth maps:

```bash
cd SFD
python depth_to_lidar.py
```

If you want to generate dense depth maps by yourself, we recommend using [TWISE](https://github.com/imransai/TWISE) or [SFD-TWISE](https://github.com/LittlePey/SFD-TWISE). Organize your dataset as follows:

```
SFD
├── data
│   ├── kitti_pseudo
│   │   │── ImageSets
│   │   │── training
│   │   │   ├── calib & velodyne & label_2 & image_2 & (optional: planes) & depth_dense_twise & depth_pseudo_rgbseguv_twise
│   │   │── testing
│   │   │   ├── calib & velodyne & image_2 & depth_dense_twise & depth_pseudo_rgbseguv_twise
│   │   │── gt_database
│   │   │── gt_database_pseudo_seguv
│   │   │── kitti_dbinfos_train_sfd_seguv.pkl
│   │   │── kitti_infos_test.pkl
│   │   │── kitti_infos_train.pkl
│   │   │── kitti_infos_trainval.pkl
│   │   │── kitti_infos_val.pkl
├── pcdet
├── tools
```

Generate KITTI dataset infos and ground-truth database:

```bash
cd tools
python -m pcdet.datasets.kitti.kitti_dataset_custom create_kitti_infos cfgs/dataset_configs/kitti_dataset_custom.yaml
```

---

## 🏆 Main Results

### KITTI Validation & Test Sets

| Dataset | Modality | mAP₃ᴰ (R40) | Easy | Moderate | Hard |
|---------|:--------:|:----------:|:----:|:--------:|:----:|
| KITTI Test | L+C | **86.12** | 92.25 | 85.47 | 80.65 |
| KITTI Val ([Checkpoint](https://huggingface.co/GGboy-ues/CoDIF/tree/main/kitti-val-pth)) | L+C | **91.09** | 96.24 | 89.85 | 87.19 |

### nuScenes Validation Set

| Method | Reference | Resolution | NDS | mAP | Params (M) | FPS |
|--------|-----------|-----------|:---:|:---:|:----------:|:---:|
| IS-Fusion | CVPR'24 | 1056×384 | 73.6 | 72.5 | 48.3 | 3.20 |
| MambaFusion-Base | ICCV'25 | 704×256 | 75.0 | 72.7 | — | 4.7 |
| **CoDIF-Light** | — | 704×256 | 73.5 | 70.7 | 18.2 | 4.18 |
| **CoDIF ([Checkpoint](https://huggingface.co/GGboy-ues/CoDIF/tree/main/nuScenes-val-test-pth))** | — | 704×256 | **75.4** | **73.3** | 35.5 | 3.06 |

### nuScenes-C Robustness Benchmark

| Method | Clean | Snow | Rain | Fog | Sunlight | Density | Cutout | Crosstalk |
|--------|:----:|:----:|:----:|:---:|:--------:|:------:|:-----:|:---------:|
| Baseline | 71.53 | 68.34 | 66.43 | 69.45 | 67.69 | 70.95 | 69.60 | 68.69 |
| **CoDIF ([Checkpoint](https://huggingface.co/GGboy-ues/CoDIF/tree/main/nuScenes-val-test-pth))** | **73.28** | **69.96** | **68.85** | **70.62** | **69.44** | **71.91** | **71.08** | **70.04** |


### Training Modes

The framework supports three operational modes controlled by `MODEL.FUSER.TRAIN_MODE`:

- **`train`**: Standard detection training with convolutional fusion only (diffusion disabled).
- **`difftrain`**: Diffusion pre-training with physically guided noise to learn the UNet-based cross-modal denoising.
- **`finetune`**: Joint end-to-end fine-tuning combining detection supervision with light diffusion regularization.

---

## Training

### NuScenes

#### Stage 1 — Detection Backbone Training

Start by downloading the [Vmamba pretrained weights](https://huggingface.co/GGboy-ues/CoDIF/tree/main/Vmamba-pretrain).

```bash
cd tools
bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/Codiff.yaml --sync_bn --pretrained_model ckpts/pretrained.pth --logger_iter_interval 100
```

#### Stage 2 — Diffusion Model Training

```bash
cd tools
python -m torch.distributed.launch --nproc_per_node=4 --master_port 25530 train_diffv1.py \
    --num_epochs 1000 --batch_size 1 --save_every 10 --lr 1e-3 \
    --save_dir ../output/DIff_checkpoints_physical
```

#### Stage 3 — Finetune

```bash
cd tools
bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/Codiff_finetune.yaml --sync_bn --pretrained_model ckpts/pretrained.pth --logger_iter_interval 500
```

#### Lightweight Variant

For the lightweight variant (CoDIF-Light), replace the config files with the `_light` versions:

```bash
cd tools
bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/Codiff_light.yaml --sync_bn --pretrained_model ckpts/pretrained.pth
```

### KITTI

All training / evaluation commands below should be run from the `tools/` directory under `CoDIF-KITTI`.

**Train CoDIF (main model) on KITTI val set:**

```bash
cd tools

CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch \
    --nnodes 1 --nproc_per_node=1 --master_port 25511 train.py \
    --gpu_id 0 --launch 'pytorch' --workers 4 \
    --batch_size 1 --cfg_file cfgs/kitti_models/mpcf_codiff.yaml \
    --tcp_port 61000 \
    --epoch 20 --max_ckpt_save_num 20 \
    --fix_random_seed
```

**Train diffusion model:**

```bash
cd tools

CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port 25530 train_diffv1.py \
    --num_epochs 1000 --batch_size 1 --save_every 5 \
    --lr 1e-4 --lr_scheduler cosine \
    --save_dir ../output/DIff_checkpoints_physical
```

**Finetune:**

```bash
cd tools

CUDA_VISIBLE_DEVICES='1' python -m torch.distributed.launch \
    --nnodes 1 --nproc_per_node=1 --master_port 25530 train.py \
    --gpu_id 1 --launch 'pytorch' --workers 8 \
    --batch_size 1 --cfg_file cfgs/kitti_models/mpcf_codiff.yaml \
    --tcp_port 61000 \
    --epoch 30 --max_ckpt_save_num 25 \
    --fix_random_seed \
    --pretrained_model ../output/kitti_models/mpcf_codiff/default/ckpt/checkpoint_epoch_53.pth
```



---

## Inference

### NuScenes

```bash
cd tools
bash scripts/dist_test.sh 3 --cfg_file cfgs/nuscenes_models/Codiff_test.yaml --ckpt path/to/checkpoint.pth
```

### KITTI

**Evaluate all checkpoints on KITTI val set:**

```bash
cd tools

CUDA_VISIBLE_DEVICES='0' python test.py \
    --gpu_id 0 --workers 4 \
    --cfg_file cfgs/kitti_models/mpcf_codiff_test.yaml \
    --batch_size 1 --eval_all
```

**Evaluate with a specific checkpoint on KITTI val set:**

```bash
cd tools

CUDA_VISIBLE_DEVICES='1' python test.py \
    --gpu_id 1 --workers 0 \
    --cfg_file cfgs/kitti_models/mpcf_codiff_test.yaml \
    --batch_size 1 \
    --ckpt ../output/kitti_models/mpcf_codiff/default/ckpt/checkpoint_epoch_14.pth
```

---

## 🗂️ Available Configurations (KITTI)

Model configs are located in `tools/cfgs/kitti_models/`:
Dataset configs are located in `tools/cfgs/dataset_configs/`:
- `kitti_dataset.yaml` — KITTI dataset config
- `kitti_dataset_custom.yaml` — Custom KITTI (with pseudo-lidar / dense depth)
- `nuscenes_dataset.yaml` — nuScenes dataset config

---

## ✨ Citation

If you find this work useful, please consider citing:

```bibtex
@article{gao2025codif,
  title={CoDIF: Conditional Diffusion Fusion with Soft Alignment for Robust Multimodal 3D Object Detection},
  author={Gao, Pan and Cheng, Jianmei and Fei, Chun and Ren, Shuai and Fan, Yuling and Zhang, Ping},
  journal={},
  year={2026}
}
```

---

## Acknowledgments

CoDIF uses code from several open source repositories. We gratefully acknowledge these contributions:

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [MambaFusion](https://github.com/AutoLab-SAI-SJTU/MambaFusion)
- [VMamba](https://github.com/MzeroMiko/VMamba)
- [VoxelMamba](https://github.com/gwenzhang/Voxel-Mamba)
- [LION](https://github.com/happinesslz/LION)
- [UniTR](https://github.com/Haiyang-W/UniTR)
