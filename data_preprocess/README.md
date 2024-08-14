# 1. Environment
## 1.1 Env variables
**CUDA**
```
# use cuda-11.7 by default since torch1.13.1 is compatible with cuda-11.7, remember to link /usr/local/cuda-11.7 to /usr/local/cuda
CUDA_HOME="/usr/local/cuda"
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64
```

**Colmap**
```
pip install colmap
```

**Python**
```
# source code for python 3.10.12: /mnt/volumes/jointmodel/bicheng/install/Python-3.10.12
# optimized building for python 3.10.12: /home/root/.envs/py3_10_12
# base virtualenv (python3.10.12+pytorch1.13.1+CUDA11.7): /mnt/volumes/jointmodel/bicheng/envs/py310/bin/python

# Create new virtualenv based on base env
virtualenv -p /home/root/.envs/py3_10_12 $SAVE_PATH

# Activate base virtualenv for LiGS
source /mnt/volumes/jointmodel/bicheng/envs/py310_ligs/bin/activate

# Deactivate virtualenv
source deactivate
```

**Pytorch&Essential Packages**
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install colorama ipython matplotlib tqdm pyyaml easydict open3d scikit-learn scikit-image imageio imageio[ffmpeg] opencv-python ninja plyfile tensorboardx
```

## 3. Pipeline
***Preparation for datasets***
```
# Preprocess progress: colmap -> segmentation -> pcd_clean -> pcd_standard -> pcd_rescale

# colmap: use colmap to extract camera intrinsics and extrinsics.
./bash/pipeline.sh DATASET_NAME dataset EXP_NAME

# segmentation: use SAM to extract alpha for car. Should be run after colmap.
./bash/pipeline.sh DATASET_NAME segmentation EXP_NAME

# pcd_clean: use alpha extracted by SAM to clean point cloud. Should be run after segmentation.
./bash/pipeline.sh DATASET_NAME pcd_clean EXP_NAME

# pcd_standard: use PCA to find possible standarized coordinates. CHECKING REQUIRED since may be wrong. Should be run after pcd_clean.
./bash/pipeline.sh DATASET_NAME pcd_standard EXP_NAME

# pcd_rescale: use PCA calculated by ARKit camera extrinsics to extract scales between colmap camera extrinsics and arkit's. Should be run after pcd_standard.
./bash/pipeline.sh DATASET_NAME pcd_rescale EXP_NAME

```