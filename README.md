### TwoStageDetector
```
@misc{CV2018,
  author =       {zhixiangduan (zhixiangduan@deepmotion.ai)},
  howpublished = {{http://gitlab.deepmotion.ai/zhixiang/TwoStageDetector}},
  year =         {2018}
}
```

### Features
* ```Pure Pytorch Code```
* ```Three Pooling Methods```: roi-pooling, roi-align, roi-crop

### Main Results
| methods           |     ap   |
|-------------------|----------|
| baseline          |   89.21% |
| better_subsample  |   89.38% |

### Set up the environment
* This repo has been test on Python3.5 and Pytorch 0.4.0

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
* modify the some paths in configs/*.json like as examples

### Configs
It can be generated by python scripts for convenience.
Of course you can modified it manually. but you need be careful
to make sure the consistency of configuration.(e.g. the same postprocess in eval and train)

There are lots of options in config files,you can just use
configs/refine_kitti_config.json for free


### Training
* To train KITTI dataset, simply run:

```bash
CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
--net iou_faster_rcnn \
--out_path /data/object/liangxiong/iou_exp \
--config configs/iou_kitti_config.json
```
* There are many options that can be used
    * --net: specify net arch
    * --out_path: all output file will put there
    * --config: config file
    * --cuda: enable gpu

* To resume from checkpoint,run
```
CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
 --net two_rpn \
 --out_path /data/object/liangxiong/two_rpn \
 --config configs/two_rpn_config.json
 --r True \
 --checkpoint 3257 \
 --checkepoch 1
```
modify the corresponding options for yourself

### Test
for evaluation model in val dataset,run
```bash
CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
--checkpoint 3257 \
--checkepoch 37 \
--net iou_faster_rcnn \
--load_dir /data/object/liangxiong/iou_exp
```
note that dont need to specify config file for test, 
the script will find it in the directory of experiments


### Visualization
After running test,result will be put in results/data/ directory.
```bash
python vis_all.py
```
It will visualize all results in order

### Evaluation
just run sh eval.sh, then you can get three 2D aps for easy,moderate,hard

### Development
* The directory structure of project is like
```
build/
    xxxx_builder.py
core/
    models/
    similarity_cals/
    bbox_coders/
    losses/
    ops/
```
* structure description
    * all models will put in models/(e.g. rpn_model.py,faster_rcnn_model.py).
    * similarity_cals/ is used for calculating similarity between bboxes and gt_boxes
    * some common pytorch operators will put in ops/(e.g. meshgrid)
    * all builder func will be put in build/

If you want to develop a custom model, inherit from core/model.py and realize some abstract functions
like ```init_params```,```init_modules``` and others

If many classes can be selected, builder design pattern can be used, do the same thing like as build/

