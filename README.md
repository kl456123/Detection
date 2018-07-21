### Source code for TwoStageDetector(Pytorch)
```
@misc{CV2018,
  author =       {zhixiangduan (zhixiangduan@deepmotion.ai)},
  howpublished = {{http://gitlab.deepmotion.ai/zhixiang/TwoStageDetector}},
  year =         {2018}
}
```

### Features
* ```Pure Pytorch Code```
* ```Supports Multi-Image Batch Training```
* ```Three Pooling Methods```: roi-pooling, roi-align, roi-crop

### Set up the environment
* This repo has been test on Python2.7 and Pytorch 0.3.0-post4

### Prepare the data
* data structure should be like this:

```
Kitti
    object
        training
            image_2
            label_2
            train.txt
            val.txt
```

* then modify data_root_path = 'path/to/training' in trainval_net.py

### Training
* To train KITTI dataset, simply run:

```bash
python trainval_net.py  --bs 4 --cuda
```

```
trainval_bev and test_bev for bev
trainval and test for fv

python trainval_net.py --cuda --net resnet50
python test_net.py --cuda --checkpoint 3257 --checkepoch 100 --load_dir /data/liangxiong/models/ --net resnet50
```
