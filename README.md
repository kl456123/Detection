
# Geometry method


## environments
* pytorch1.0
* python3.5

## Prepare
### compile
```bash
# compile lib
cd lib
python setup.py build develop
cd -
```

### dataset
download kitti, list all files as following
- object/training
 - image_2
 - calib
 - label_2


### configuration
* data_config, use type to select different datasets
    - dataloader_config, config torch.data.dataloader
    - dataset_config
        * classes, Car or Pedestrian
        * root_path, root path of dataset
        * dataset_file, list of sample names(for ex 000000, 000001 ...)
        * use_proj_2d, predict boxes_2d or projection of 3d bbox(more hard to match proj with anchors)
    - transform_config, more details of data augumentations are in data/transforms/mono_kitti_transform.py
        * crop_size, resize
        * resize_range, random crop image
* eval_config
    * used_to configurating tester(details in core.tester)
* eval_data_config, like data_config, just used when evaluation
* model_config, used for config models(the top model, like faster rcnn)
    - rpn_config, config of submodule
    - feature_extractor_config, backbone
    - sampler_config, subsample bboxes to reduce number of inputs
    - target_assigner_config, generate targets to calc loss for each proposal
        - similiarity_calc_config, refers to calc iou here
        - coder_config, encode boxes_2d(details in core.bboxes_coders)
        - coder_3d_config, encode boxes_3d(details in core.bboxes_coders)
        - matcher_config, assign gt to proposals(core.matchers)
* train_config
    - optimizer_config, sgd, adam or others
    - scheduler_config, lr_step, exp or others

**Note**
you should modify `root_path` before running scripts, if you want to use new data split file, you should also modify `dataset_file`.
Dont forget to do the same thing in `eval_data_config`.



## Training
download pretrained model(res50, res18_pruned) from `smb://deepmotion1/public/liangxiong` first
```bash
# NET=geometry_v2
# NET=geometry_v1
# NET=geometry_v3

CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    --net ${NET} \
    --out_path /data/object/liangxiong/test \
    --config configs/${NET}.json
#--model ${custom_pretrained_model_path}
```
you need to check run.sh first then run `sh run.sh` after you uncomment the NET.
Of course you can use `--model` option to load your trained model to refine

* if you need to resume model, just append resume option `--r`
```bash
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    --net ${NET} \
    --out_path /data/object/liangxiong/test \
    --config configs/${NET}.json \
    --r True \
    --checkepoch 100 \
    --checkpoint 12
```


## Testing

```bash
mkdir results/data -p

# two methods to test your model
# evaluate models you trained by yourself
# modify checkpoint and checkepoch before running it
NET=geometry_v1
DIR=''
CHECKEPOCH=30
CHECKPOINT=10
CONFIG=config/${NET}.json
MODEL=xxx.pth
CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \ 
    --net ${NET} \
    --load_dir ${DIR} \
    --checkpoint ${CHECKPOINT} \
    --checkepoch ${CHECKEPOCH}

# evaluate any models given model_path and config_path
CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    --net ${NET} \
    --config ${CONFIG}\
    --model ${MODEL}
#    --img_dir /data/dm202_3w/left_img \
#    --calib_file ./000004.txt
```
two methods are given to evaluate models. BTW if you want to inference images in some directory, append `--img_dir`, `--calib_file` options
results will saved to `results/data`


## Visualization

* modify the path of pred results in vis_all_3d.py, you also need to modify label path if you want to compare results with gt labels

```bash
# old vis scripts
# mkdir results/fv
# python vis_all_3d.py


# new visualizer tools
# path to the images
image_dir = '/data/object/training/image_2'
# path to the prediction results(kitti format)
result_dir = './results/data'
# path to the output images
save_dir = 'results/images'
# calib path for camera, each image has different calib file
calib_dir = '/data/object/training/calib'
# one calib file for all images
calib_file = '00000.txt'

# label path to compare gt and pred
#label_dir = '/data/object/training/label_2'

mkdir results/images
python utils/drawer.py
```
