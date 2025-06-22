# FSHNet
## ["FSHNet: Fully Sparse Hybrid Network for 3D Object Detection"](https://arxiv.org/abs/2506.03714)
Our paper is accepted by CVPR2025. Thanks for the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), this implementation of the DCDet is mainly based on the pcdet v0.6.

Abstract: Fully sparse 3D detectors have recently gained significant attention due to their efficiency in long-range detection. However, sparse 3D detectors extract features only from non-empty voxels, which impairs long-range interactions and causes the center feature missing. The former weakens the feature extraction capability, while the latter hinders network optimization. To address these challenges, we introduce the Fully Sparse Hybrid Network (FSHNet). FSHNet incorporates a proposed SlotFormer block to enhance the long-range feature extraction capability of existing sparse encoders. The SlotFormer divides sparse voxels using a slot partition approach, which, compared to traditional window partition, provides a larger receptive field. Additionally, we propose a dynamic sparse label assignment strategy to deeply optimize the network by providing more high-quality positive samples. To further enhance performance, we introduce a sparse upsampling module to refine downsampled voxels, preserving fine-grained details crucial for detecting small objects. Extensive experiments on the Waymo, nuScenes, and Argoverse2 benchmarks demonstrate the effectiveness of FSHNet.

### 1. Recommended Environment

- Linux (tested on Ubuntu 20.04)
- Python 3.6+
- PyTorch 1.1 or higher (tested on PyTorch 1.13)
- CUDA 9.0 or higher (tested on 11.6)

### 2. Set the Environment
```shell
pip install -r requirement.txt
python setup.py develop
```

### 3. Data Preparation
- [Preparation of different datasets.](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#dataset-preparation)


### 4. Train

- Train with a single GPU

```shell
python tools/train.py --cfg_file ${CONFIG_FILE}

# e.g.,
python tools/train.py --cfg_file tools/cfgs/waymo_models/fshnet_base.yaml
```

- Train with multiple GPUs or multiple machines

```shell
bash tools/scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
# or 
bash tools/scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# e.g.,
bash tools/scripts/dist_train.sh 8 --cfg_file tools/cfgs/waymo_models/fshnet_base.yaml
```

### 5. Test

- Test with a pretrained model:

```shell
python tools/test.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}

# e.g., 
python tools/test.py --cfg_file tools/cfgs/waymo_models/fshnet_base.yaml --ckpt {path}
```

## News
- [25-06-04] Release the [arXiv]((https://arxiv.org/abs/2506.03714)) version.
- [25-06-07] Release the code on Waymo Open dataset.

## Main results

### Waymo Open Dataset validation
|  Model  | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Log |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  [FSHNet(base)](tools/cfgs/fshnet_models/fshnet_base.yaml) |  83.0/80.8  | 77.1/74.9  | 82.3/81.9 | 74.5/74.0 | 86.2/81.1 | 79.2/74.2 | 80.4/79.3 | 77.6/76.5 | [Log](output/train_fshnet_base_12e_50.log) |

## Acknowledgement
FSHNet is greatly inspired by the following outstanding contributions to the open-source community:</p>
<ul>
    <a href="https://github.com/skyhehe123/ScatterFormer" target="_blank">ScatterFormer</a> | <a href="https://github.com/Haiyang-W/DSVT" target="_blank">DSVT</a> 
</ul>

## Paper

Please cite our paper if you find our work useful for your research:

```
@InProceedings{Liu_2025_CVPR,
    author    = {Liu, Shuai and Cui, Mingyue and Li, Boyang and Liang, Quanmin and Hong, Tinghe and Huang, Kai and Shan, Yunxiao and Huang, Kai},
    title     = {FSHNet: Fully Sparse Hybrid Network for 3D Object Detection},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {8900-8909}
}
```
