# Training Detection Model for YOLOv5

## Dataset Preparation

```
dataset/
├── images/
│   ├── train/ (training images)
│   ├── val/   (validation images)
├── labels/
│   ├── train/ (training annotations)
│   ├── val/   (validation annotations)
```

- Image format: JPG or PNG
- Label format: `class_id x_center y_center width height`
- All values are normalized (e.g., x_center is relative to the image width).

Dataset Configuration File

```yaml
train: /path/to/dataset/images/train
val: /path/to/dataset/images/val

nc: 2 # number of classes
names: ["class1", "class2"] # class names
```

## Execute Training

```bash
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt
```

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --weights yolov5s.pt --device 0,1
```

## Monitoring

```bash
tensorboard --logdir runs/train
```
