
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


## Testing

```bash
mkdir results/data -p

# two methods to test your model
# evaluate models you trained by yourself
# modify checkpoint and checkepoch before running it
CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \ 
    --net geometry_v1 \
    --load_dir /data/object/liangxiong/coco_pretrained_normalized_refine \
    --checkpoint 4912 \
    --checkepoch 49
    

# evaluate any models given model_path and config_path
CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    --net geometry_v1 \
    --config configs/geometry_v1.json\
    --model /data/object/liangxiong/mono_3d_final_plus/mono_3d_final_plus/kitti/faster_rcnn_30_1518.pth
#    --img_dir /data/dm202_3w/left_img \
#    --calib_file ./000004.txt
```
two methods are given to evaluate models. BTW if you want to inference images in some directory, append `--img_dir`, `--calib_file` options
results will saved to `results/data`


## Visualization

* modify the path of pred results in vis_all_3d.py, you also need to modify label path if you want to compare results with gt labels

```bash
mkdir results/fv
python vis_all_3d.py
```
