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

![CoDIF Overview](assets/main.png)

---

## 🔥 News

- **[2026.06]** Code of CoDIF is released.

---

## Installation

If you encounter any problems, please consult the [install.md](docs/install.md) file for the exact version requirements of each package. If the issue still isn’t resolved, feel free to open an issue.

```bash
conda create -n codif python=3.8

# You can install CUDA 11.8 (if it isn’t already on your system) by running:
# conda install --channel "nvidia/label/cuda-11.8.0" cuda
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install extra dependency
pip install -r requirements.txt

pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp38-cp38-linux_x86_64.whl

# Install nuscenes-devkit
pip install nuscenes-devkit==1.0.5 torchinfo

# Develop
python setup.py develop
python mambafusion_setup.py develop

cd selective_scan
python setup.py develop

cd mamba_diffv/mamba
python setup.py develop

python -m pip install causal-conv1d==1.2.0.post2
```

## Dataset Preparation

- Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and organize the downloaded files as follows:

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
    # --share_memory  # if using shared memory for LiDAR and image GT sampling (~24G+143G or 12G+72G)
# Shared memory greatly improves training speed but needs ~150G or ~75G extra cache memory.
# NOTE: All experiments used shared memory. Shared memory does NOT affect performance.
```

- The format of the generated data is as follows:

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

---

## 🏆 Main Results

### nuScenes Validation Set

Comparison with state-of-the-art methods on the nuScenes validation set. FPS measured on a single RTX 3090.

![nuScenes Results](assets/results.png)

| Method | Reference | Resolution | NDS | mAP | Params (M) | FPS |
|--------|-----------|-----------|:---:|:---:|:----------:|:---:|
| TransFusion | CVPR’22 | 800×448 | 71.7 | 68.9 | 37.0 | 6.51 |
| BEVFusion (MIT) | ICRA’23 | 704×256 | 71.4 | 68.5 | 40.8 | 4.73 |
| DeepInteraction | NeurIPS’22 | 800×448 | 72.6 | 69.9 | 57.9 | 1.86 |
| SparseFusion | ICCV’23 | 704×256 | 72.8 | 70.5 | 40.2 | 4.38 |
| UniTR | ICCV’23 | 704×256 | 73.3 | 70.5 | 15.6 | 4.50 |
| CMT-VoV | ICCV’23 | 1600×640 | 72.9 | 70.3 | 86.7 | 3.48 |
| DAL-Large | ECCV’24 | 1056×384 | 74.0 | 71.5 | 47.8 | 6.10 |
| IS-Fusion | CVPR’24 | 1056×384 | 73.6 | 72.5 | 48.3 | 3.20 |
| SparseLiF | ECCV’24 | 1600×640 | 74.6 | 71.2 | — | 2.9 |
| MambaFusion-Base | ICCV’25 | 704×256 | 75.0 | 72.7 | — | 4.7 |
| **CoDIF-Light (Ours)** | — | 704×256 | **73.5** | **70.7** | **18.2** | **4.18** |
| **CoDIF (Ours)** | — | 704×256 | **75.4** | **73.3** | **35.5** | **3.06** |

### KITTI Validation & Test Sets

| Dataset | Modality | mAP₃ᴰ (R40) | Easy | Moderate | Hard |
|---------|:--------:|:----------:|:----:|:--------:|:----:|
| KITTI Test | L+C | **86.12** | 92.25 | 85.47 | 80.65 |
| KITTI Val | L+C | **91.09** | 96.24 | 89.85 | 87.19 |

### nuScenes-C Robustness Benchmark

CoDIF significantly outperforms baselines under sensor perturbations (misalignment, weather degradation, density reduction), demonstrating that soft alignment provides meaningful robustness gains where hard-alignment methods degrade substantially.

---

## Method Overview

CoDIF comprises three core components:

### 1. Diffusion-driven Soft Fusion (DSF)
DSF reformulates cross-modal alignment as a conditional denoising process. It learns to establish flexible correspondences between LiDAR and camera BEV features through a learnable UNet conditioned on the target modality, replacing rigid calibration-based projection. Physical sensor degradation patterns (vibration, calibration bias, thermal drift) are injected into the training noise schedule.

### 2. Patch Context Encoder (PCE)
PCE employs a DINO-style transformer to extract global scene context from the fused BEV features. A learnable context token interacts with all image patches through self-attention, capturing holistic scene semantics that guide local feature interpretation — especially valuable when local features are displaced by misalignment.

### 3. Global Context Refinement (GCR) with Confidence Gating
GCR applies Mamba-based multi-directional scanning (four scan directions) to capture long-range spatial dependencies. A hierarchical confidence gating mechanism operates at both intra-modality (pixel-level residual gate) and inter-modality (modality-level fusion gate) levels, adaptively weighting contributions based on per-pixel uncertainty.

### Training Modes

The framework supports three operational modes controlled by `MODEL.FUSER.TRAIN_MODE`:

- **`train`**: Standard detection training with convolutional fusion only (diffusion disabled).
- **`difftrain`**: Diffusion pre-training with physically guided noise to learn the UNet-based cross-modal denoising.
- **`finetune`**: Joint end-to-end fine-tuning combining detection supervision with light diffusion regularization.

---

## Training

### Stage 1 — Detection Backbone Training

Start by downloading the [pretrained weights](https://drive.google.com/drive/folders/1TqvpIHA7plzoFdnGWvFgVYr45bgz-nQ3?usp=sharing).

```bash
cd tools
bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/Codiff.yaml --sync_bn --pretrained_model ckpts/pretrained.pth --logger_iter_interval 100
```

### Stage 2 — Diffusion Model Training

```bash
cd tools
python -m torch.distributed.launch --nproc_per_node=4 --master_port 25530 train_diffv1.py \
    --num_epochs 1000 --batch_size 1 --save_every 10 --lr 1e-3 \
    --save_dir ../output/DIff_checkpoints_physical
```

### Stage 3 — Finetune

```bash
cd tools
bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/Codiff_finetune.yaml --sync_bn --pretrained_model ckpts/pretrained.pth --logger_iter_interval 500
```

### Lightweight Variant

For the lightweight variant (CoDIF-Light), replace the config files with the `_light` versions:

```bash
cd tools
bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/Codiff_light.yaml --sync_bn --pretrained_model ckpts/pretrained.pth
```

---

## Inference

```bash
cd tools
bash scripts/dist_test.sh 3 --cfg_file cfgs/nuscenes_models/Codiff_test.yaml --ckpt path/to/checkpoint.pth
```

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

