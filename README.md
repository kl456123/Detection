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


### Usage
* To train KITTI dataset, simply run:

```bash
python main.py  trainval_net.py  --bs 4 --cuda
```