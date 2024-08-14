dataset_name=$1
exp_name=$2

# for cmd in pcd_clean pcd_standard pcd_rescale train infer export; do
for cmd in dataset segmentation pcd_clean pcd_standard pcd_rescale; do
# for cmd in pcd_standard pcd_rescale train infer export; do
    echo "Starting progress ${cmd} ..."
    ./bash/pipeline.sh ${dataset_name} ${cmd} ${exp_name}
    echo "Done"
done

