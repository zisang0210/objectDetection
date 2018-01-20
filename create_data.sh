#!/bin/bash
# 查找脚本所在路径，并进入
#DIR="$( cd "$( dirname "$0"  )" && pwd  )"
DIR=$PWD
cd $DIR
echo current dir is $PWD

# 设置目录，避免module找不到的问题
export PYTHONPATH=$PYTHONPATH:$DIR:$DIR/slim:$DIR/object_detection

python /home/zisang/objDetect/object_detection/dataset_tools/create_data.py \
    --label_map_path=/home/zisang/data/labels_items.txt \
    --data_dir=/home/zisang/objDetect/data \
    --validation_set_size=20
