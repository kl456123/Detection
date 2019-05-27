# Benchmark

## Mono 3D

### Main results

| methods           |   dataset | speed(fps) |    ap@0.7   |     ap@0.5    | Download |
|-------------------|----------|-------------------|----------|----------|-----------|
| geometry constrain | kitti   | ~6        |    ~6        |    ~23 |   [xxx]()     |
| OFT  |  kitti  |  xxx   |   xxx    |   10    |   [xxx]()     |
| keypoint 2d  | kitti |  ~6    |    xxx       |    xxx         |   [xxx]()     |


### Scripts
```bash
# keypoint 2d (use plane prediction)
# training
CUDA_VISIBLE_DEVICES=0 python train.py --cuda \
    --net fpn_corners_2d \
    --out_path /data/object/liangxiong/test \
    --config configs/fpn_corners_2d_mono_3d_kitti_config.json \
    --model /data/object/liangxiong/test/fpn_corners_3d/mono_3d_kitti/detector_600000.pth
# Note that to use --model option to load pretrained model
# (here to load pretrained 2d car detection model)
# And if you want to use multigpus just add more gpu ids
# to CUDA_VISIBLE_DEVICES and add --mGPUs option

# inference(checkpoint number)
CUDA_VISIBLE_DEVICES=1 python test.py --cuda \
    --checkpoint 600000 \
    --load_dir /data/object/liangxiong/test \
    --net fpn_corners_2d \
    --thresh 0.5 \
    --dataset nuscenes \
    --img_dir /data/dm202_3w/left_img \
    --calib_file ./000004.txt
# --calib_file refers to single calib file
# --calib_dir refers to directory of calibs(calib format is like that of kitti)
# --img_dir refers to directory where you want to infer

# inference(model path)
CUDA_VISIBLE_DEVICES=1 python test.py --cuda \
    --model ./faster_rcnn_32_3257.pth \
    --config ./configs/refine_kitti_config.json
    --net fpn_corners_2d \
    --thresh 0.5 \
    --dataset nuscenes
# note that if no dir of file is specified, use the val dataset to infer
# In the following snippet, the code of inference is omited.
```


```bash
# geometry constrain
CUDA_VISIBLE_DEVICES=0 python train.py --cuda \
    --net fpn_mono_3d \
    --out_path /data/object/liangxiong/test \
    --config configs/fpn_mono_3d_kitti_config.json \
    --model /data/object/liangxiong/fpn_bdd_pretrained/fpn/bdd/detector_300000.pth
```

```bash
# OFT

```




## 2D

### Main results
| methods           |   dataset | speed(fps) |    ap@0.7   | Download |
|-------------------|----------|-------------------|----------|-------|
| fpn_faster_rcnn | kitti   | x        |    ~86        | [xxx]()  |
| faster_rcnn  |  kitti  |  xxx   |   xxx    | [xxx]()  |
| faster_rcnn | coco |  ~6    |    xxx       | [xxx]()  |
| faster_rcnn | bdd |  ~6    |    xxx       | [xxx]()  |
| faster_rcnn | nuscenes |  ~6    |    xxx       | [xxx]()  |
| ssd | nuscenes |  xx    |    xxx       | [xxx]()  |
| prnet | nuscenes |  xx    |    xxx       | [xxx]()  |

### Scripts

```bash
# KITTI
CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    --net fpn \
    --out_path /data/object/liangxiong/fpn_kitti_pretrained \
    --config configs/fpn_kitti_config.json
```


```bash
# BDD
CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    --net fpn \
    --out_path /data/object/liangxiong/fpn_bdd_pretrained \
    --config configs/fpn_bdd_config.json

```

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    --net fpn \
    --out_path /data/object/liangxiong/fpn_coco_pretrained \
    --config configs/fpn_coco_config.json
```


Note that all model use the same backbone(res18_pruned)


## PointCloud







## ToolKits

### Automatic generate config

```bash

python utils/generate_configs.py

```
If you just want to test the correction of your algorithm, enable DEBUG mode in file


### Visualization

```bash

python utils/drawer.py

```
For each different dataset, just to uncomment the counterpart configs is Ok








