# source ~/.bashrc
dataset_name=$1
command=$2
exp_name=$3 
if [ -z ${dataset_name} ]; then
    echo "Must give dataset_name"
    exit 0;
fi
if [ -z ${command} ]; then
    echo "Must give command"
    exit 0;
fi
if [ -z ${exp_name} ]; then
    echo "Must give exp_name"
    exit 0;
fi
# RDMA
dataset_dir=/path/to/save/path
codebase_dir=/path/to/3DRealCar_Dataset/data_preprocess
cd ${codebase_dir}

# currently we only use colmap data
processed_type=colmap
processed_dataset_dir=${dataset_dir}/${dataset_name}/${processed_type}_processed

# change to your own yaml
yaml_fn=resources/configs/${exp_name}.yaml

################################################################################
# Get processed type
################################################################################
pcd_clean_dir=pcd_clean
pcd_standard_dir=pcd_standard
pcd_rescale_dir=pcd_rescale
final_processed_dir=""
if [ -f ${processed_dataset_dir}/.dataset ]; then
    final_processed_dir=${processed_dataset_dir}
fi
if [ -f ${processed_dataset_dir}/${pcd_clean_dir}/.processed ]; then
    final_processed_dir=${pcd_clean_dir}
fi
if [ -f ${processed_dataset_dir}/${pcd_standard_dir}/.processed ]; then
    final_processed_dir=${pcd_standard_dir}
fi
if [ -f ${processed_dataset_dir}/${pcd_rescale_dir}/.processed ]; then
    final_processed_dir=${pcd_rescale_dir}
fi
echo "Processed type=${final_processed_dir}"

################################################################################
# Convert dataset from 3Dscanner to COLMAP
################################################################################
if [ $command == 'dataset' ]; then
    if [ -f ${processed_dataset_dir}/.dataset ]; then
        echo "Skip ${command} since has been already processed!"
        exit 0;
    fi
    if [ ! -d ${dataset_dir}/${dataset_name}/3dscanner_origin ]; then
        mkdir -p ${dataset_dir}/${dataset_name}
        ln -s ${dataset_dir}/${dataset_name}/3dscanner_origin ${dataset_dir}/${dataset_name}/3dscanner_origin
    fi
    python3 entrances/dataset_adaptor.py ${processed_type} \
        --search_dir ${dataset_dir}/${dataset_name}/3dscanner_origin \
        --save_dir ${processed_dataset_dir}
    mkdir -p ${processed_dataset_dir}/arkit
    cp -r ${dataset_dir}/${dataset_name}/3dscanner_origin/frame*.json ${processed_dataset_dir}/arkit/
    echo "copied arkit jsons"
    exit 0;
fi


################################################################################
# Extract semantic segmentation
################################################################################

max_attempts=1

# TODO: should fix bug when alpha channel is incorrect!
if [ $command == 'segmentation' ]; then
    if [ ! -f ${processed_dataset_dir}/.dataset ]; then
        echo "Call dataset first!"
        exit 0;
    fi
    if [ -f ${processed_dataset_dir}/.segmentation ]; then
        echo "Skip ${command} since has been already processed!"
        exit 0;
    fi
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        # Sometimes docker may not download SAM models from huggingface
        python3 utils/toolkit/extract_segmentation.py \
            --yaml ${yaml_fn} \
            --dataset_dir ${processed_dataset_dir}
        if [ -f ${processed_dataset_dir}/.segmentation ]; then
            exit 0;
        fi
        echo "Attempt ${attempt} failed."
    done
    echo "Segmentation failed after ${max_attempts} attempts."
    exit 0;
fi

################################################################################
# PCD Cleaning
################################################################################
if [ $command == 'pcd_clean' ]; then
    if [ ! -f ${processed_dataset_dir}/.segmentation ]; then
        echo "Call segmentation first!"
        exit 0;
    fi
    if [ -f ${processed_dataset_dir}/${pcd_clean_dir}/.processed ]; then
        echo "Skip ${command} since has been already processed!"
        exit 0;
    fi
    python3 utils/toolkit/extract_foreground_pcd.py \
        --yaml ${yaml_fn} \
        --dataset_dir ${processed_dataset_dir} \
        --save_dir ${processed_dataset_dir}/${pcd_clean_dir}
    python3 utils/toolkit/visualize_dataset.py \
        --yaml ${yaml_fn} \
        --dataset_dir ${processed_dataset_dir}/${pcd_clean_dir} \
        --save_dir ${processed_dataset_dir}/${pcd_clean_dir}
    exit 0;
fi

################################################################################
# Standardize Coordinates
################################################################################
if [ $command == 'pcd_standard' ]; then
    if [ ! -f ${processed_dataset_dir}/${pcd_clean_dir}/.processed ]; then
        echo "Call pcd_clean first!"
        exit 0;
    fi
    if [ -f ${processed_dataset_dir}/${pcd_standard_dir}/.processed ]; then
        echo "Skip ${command} since has been already processed!"
        exit 0;
    fi
    python3 utils/toolkit/standarize_coordinates.py \
        --yaml ${yaml_fn} \
        --dataset_dir ${processed_dataset_dir}/${pcd_clean_dir} \
        --save_dir ${processed_dataset_dir}/${pcd_standard_dir} \
        --dataset_name ${dataset_name} \
        --manual_setting resources/pcd_standard.txt
    cp ${processed_dataset_dir}/${pcd_clean_dir}/trainval.meta ${processed_dataset_dir}/${pcd_standard_dir}/
    python3 utils/toolkit/visualize_dataset.py \
        --yaml ${yaml_fn} \
        --dataset_dir ${processed_dataset_dir}/${pcd_standard_dir} \
        --save_dir ${processed_dataset_dir}/${pcd_standard_dir}
    exit 0;
fi

################################################################################
# Rescale Coordinates
################################################################################
if [ $command == 'pcd_rescale' ]; then
    if [ ! -f ${processed_dataset_dir}/${pcd_standard_dir}/.processed ]; then
        echo "Call pcd_standard first!"
        exit 0;
    fi
    if [ -f ${processed_dataset_dir}/${pcd_rescale_dir}/.processed ]; then
        echo "Skip ${command} since has been already processed!"
        exit 0;
    fi
    python3 utils/toolkit/rescale_colmap.py \
        --yaml ${yaml_fn} \
        --dataset_dir ${processed_dataset_dir}/${pcd_standard_dir} \
        --save_dir ${processed_dataset_dir}/${pcd_rescale_dir}
    cp ${processed_dataset_dir}/${pcd_standard_dir}/trainval.meta ${processed_dataset_dir}/${pcd_rescale_dir}/
    python3 utils/toolkit/visualize_dataset.py \
        --yaml ${yaml_fn} \
        --dataset_dir ${processed_dataset_dir}/${pcd_rescale_dir} \
        --save_dir ${processed_dataset_dir}/${pcd_rescale_dir}
    touch ${processed_dataset_dir}/.processed
    exit 0;
fi
