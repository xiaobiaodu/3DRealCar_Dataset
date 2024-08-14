# This script is mainly used in utils/adaptor/dataset.py
# The main purpose of this script is to wrap colmap into python classes
search_dir=$1
save_dir=$2
image_dir=${save_dir}/input
mkdir -p ${image_dir}

echo "3dscanner_to_colmap, search_dir=${search_dir}, save_dir=${save_dir}"
cp -r ${search_dir}/frame*.jpg ${image_dir}
num_images=`ls ${image_dir}/*.jpg | wc -l`
if [ ${num_images} -le 0 ]; then
    echo "NO images are found!"
    exit 0;
fi
python3 utils/toolkit/convert_colmap.py -s ${save_dir}

