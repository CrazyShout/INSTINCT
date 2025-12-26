# INSTINCT (ICCV 2025)

INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception 

![INSTINCT](images/INSTINCT.jpg)

## Abstract
Collaborative perception systems overcome single-vehicle limitations in long-range detection and occlusion scenarios by integrating multi-agent sensory data, improving accuracy and safety. However, frequent cooperative interactions and real-time requirements impose stringent bandwidth constraints. Previous works proves that query-based instance-level interaction reduces bandwidth demands and manual priors, however, LiDAR-focused implementations in collaborative perception remain underdeveloped, with performance still trailing state-of-the-art approaches. To bridge this gap, we propose INSTINCT (instance-level interaction architecture), a novel collaborative perception framework featuring three core components:  1. a quality-aware filtering mechanism for high-quality instance feature selection;  2. a dual-branch detection routing scheme to decouple collaboration-irrelevant and collaboration-relevant instances;  3. a Cross Agent Local Instance Fusion module to aggregate local hybrid instance features.  Additionally, we enhance the ground truth (GT) sampling technique to facilitate training with diverse hybrid instance features. Extensive experiments across multiple datasets demonstrate that INSTINCT achieves superior performance. Specifically, our method achieves an improvement in accuracy 13.23%/32.24% in DAIR-V2X and V2V4Real while reducing the communication bandwidth to 1/281 and 1/264 compared to state-of-the-art methods. The code will be released soon.

## Installation

First, clone the repo:
```
git clone https://github.com/CrazyShout/INSTINCT.git
```

Next we create a conda environment and install the requirements.
```
conda create -n instinct python=3.7.16
activate instinct
```

Install the requirements:
```
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --index-url https://download.pytorch.org/whl/cu116
pip install spconv-cu116
pip install -r requirements.txt
```

Build and install other packages:
```
python setup.py develop

# gpu is required
python opencood/utils/setup.py build_ext --inplace

python opencood/pcdet_utils/setup.py build_ext --inplace 
```
if you meet problems while building pcdet_utils, please try to comment following codes in `opencood\pcdet_utils\pointnet2`, `opencood\pcdet_utils\roiaware_pool3d` and `opencood\pcdet_utils\iou3d_nms`:
```
#include <THC/THC.h>
......
extern THCState *state;

# comment all of them
// #include <THC/THC.h>
......
// extern THCState *state;
```
Install OpenPCDet
```
# OpenPCDet is required.
git clone https://github.com/open-mmlab/OpenPCDet.git
cd ../OpenPCDet
python setup.py develop
```


## Data Preparation

The data preparation is also the same as that of CoAlign and [OpenCOOD](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html). For the DAIR-V2X dataset, please use the [supplemented annotations](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/).


## Quick Start

### Train the model.
To quickly train your own INSTINCT, please run the following commond:
```
python opencood/tools/train_simple.py -y dairv2x opencood/lidar_only_with_noise/second_CQCPInstance_onecycle.yaml
```

### Test the model.
Suppose your trained INSTINCT is saved in `#your_INSTINCT_path`, and then run this command:

```
python opencood/tools/inference_simple.py --model_dir #your_INSTINCT_path
```

## Checkpoints

The main checkpoints can be downloaded [coming soon](https://drive.google.com), and then save them in the `opencood/logs` directory. Note that our checkpoints rely on `spconv=2.3.6`.

## Citation


## Acknowledgements

Thank for the excellent collaborative perception codebases [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) , [CoAlign](https://github.com/yifanlu0227/CoAlign) and [ConQueR](https://github.com/V2AI/EFG).

