
# Geometry method


## results



## environments
* pytorch1.0
* python3.5


## Training
```bash
CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    --net mono_3d_final_plus \
    --out_path /data/object/liangxiong/coco_pretrained_normalized_refine \
    --config configs/coco_mono_3d_config.json
```
you need to check run.sh first then run `sh run.sh`


## Testing

```bash
mkdir results/data -p

CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \ 
    --net mono_3d_final_plus \
    --load_dir /data/object/liangxiong/coco_pretrained_normalized_refine \
    --checkpoint 4912 \
    --checkepoch 49
    #    --img_dir /data/dm202_3w/left_img \
    #    --calib_file ./000004.txt
```
results will saved to `results/data`


## Visualization

```bash
mkdir results/fv
python vis_all_3d.py
```
