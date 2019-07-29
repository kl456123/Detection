# Benchmark and Model Zoo


## Environment

### hardware
- single 1080Ti(only single batch is supported for this codebase)

### Software environment

- Python 3.5 / 3.6
- Pytorch 1.0
- CUDA8.0


## Common setting
- use res50 as backbone, just extend faster rcnn framework to 3d misson by using add new head on the top of roi features
- optimizer is the same for all models, lr=0.001, lr_decay_gamma=0.1 lr_decay_step=50000, if so after 30epoch the model may be work in most case
- pretrained in imagenet



## Baselines

### geometry constraints

|  Style        |  Mem (GB) | Train time (s/iter) | Inf time (fps) | AP@0.7             |                                                          Download                                                          |
| :-----------: |  :------: | :-----------------: | :------------: | :-------------:    | :------------------------------------------------------------------------------------------------------------------------: |
|  geometry_v1  |   3021    |          0.28         |      6~7      |  11.03(30 epoch)  |      [model](smb://deepmotion1/public/liangxiong/models_zoo/geometry_v1_30_1518.pth)       |
|  geometry_v2  |   3003    |          0.28         |      6~7     |   ~13              |      [model]()       |
|  geometry_v3  |   3003    |          0.28         |      6~7     |   ~12.48            |      [model]()      |


**Notes:**
- geometry_v1: 估计３ｄ投影,然后编码侧边线，预测维度，最后优化位置
- geometry_v2: 去除优化位置步骤，直接估深度，中心投影代替
- geometry_v3:　估计３ｄ投影换成２ｄｂｂｏｘ(主要解决如何编码侧边线问题，可能侧边线投影数值很大，遮挡看不到等)

