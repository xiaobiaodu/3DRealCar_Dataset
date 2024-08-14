
DATA_ROOT=$1
SAVE_DIR=$2
WIDTH=$3
HEIGHT=$4

python utils/toolkit/depth_extraction/colmap2mvsnet.py \
    --dense_folder $DATA_ROOT \
    --max_d 256 \
    --save_dir ${SAVE_DIR}

python utils/toolkit/depth_extraction/vismvsnet_test.py \
    --data_root $SAVE_DIR \
    --resize 1280,720 \
    --crop 1280,720 \
    --load_path submodules/Relightable3DGaussian/vismvsnet/pretrained_model/vis

python utils/toolkit/depth_extraction/vismvsnet_filter.py \
    --data $SAVE_DIR/vis_mvsnet \
    --pair $SAVE_DIR/pair.txt \
    --vthresh 2 \
    --pthresh '.6,.6,.6' \
    --out_dir $SAVE_DIR/filtered \
    --out_shape ${WIDTH},${HEIGHT}


