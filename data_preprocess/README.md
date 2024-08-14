**install**
**install**
```
# install colmap
apt-get install colmap

# install some python package
pip install colorama plyfile open3d kornia tqdm imageio imageio[ffmpeg] opencv-python

# other you need install GroudingDino and SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && mv sam_vit_h_4b8939.pth resources/models/
pip install submodules/GroundingDINO
pip install supervision segment_anything
```

**data**
```
# data path structure example: 
/path/raw_data/dataset_name/3dscanner_origin/xxxx(png or json)
# for example: /realcar3D/demo_data/raw_data/2024_04_11_11_31_27/3dscanner_origin
```

**run**
```
# for example, DATASET_NAME=2024_04_11_11_31_27, EXP_NAME=demo

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