export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/dataset_tools/create_pascal_split_tf_record.py \
     --label_map_path=object_detection/data/pascal_label_map.pbtxt \
     --data_dir=VOCdevkit --year=VOC2012 --set=train
python object_detection/dataset_tools/create_pascal_split_tf_record.py \
     --label_map_path=object_detection/data/pascal_label_map.pbtxt \
     --data_dir=VOCdevkit --year=VOC2012 --set=val

python object_detection/train.py \
	 --train_dir=/home/zisang/objDetect/models/yolo/train \
	 --pipeline_config_path=/home/zisang/objDetect/object_detection/samples/configs/yolo_googlenet_voc.config

python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=/home/zisang/objDetect/object_detection/samples/configs/yolo_googlenet_voc.config \
    --checkpoint_dir=/home/zisang/objDetect/models/yolo/train \
    --eval_dir=/home/zisang/objDetect/models/yolo/eval
# 导出模型
python ./object_detection/export_inference_graph.py \
	--input_type image_tensor \
	--pipeline_config_path /home/zisang/objDetect/object_detection/samples/configs/yolo_googlenet_voc.config \
	--trained_checkpoint_prefix /home/zisang/objDetect/models/yolo/train/model.ckpt-711 \
	--output_directory /home/zisang/objDetect/models/yolo/exported_graphs

# 在test.jpg上验证导出的模型
python ./inference.py --output_dir=/home/zisang/objDetect/models/yolo \
	--dataset_dir=/home/zisang/objDetect/models/yolo
protoc object_detection/protos/yolo.proto object_detection/protos/model.proto --python_out=.
tensorboard --logdir=models/yolo/eval
find . -name '__pycache__' -type d -exec rm -rf {} \;