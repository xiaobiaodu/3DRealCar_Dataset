image_dir=$1
video_fn=$2
fps=$3
prefix=$4

if [ -z ${prefix} ]; then
    prefix=%d
fi

ffmpeg -y \
    -loglevel error \
    -r ${fps} \
    -i ${image_dir}/${prefix}.jpg \
    -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" \
    ${video_fn}


