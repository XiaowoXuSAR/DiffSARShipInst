# DiffSARShipInst
**Installation**

The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [Sparse R-CNN](https://github.com/facebookresearch/detectron2), [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), and [DiffusionInst](https://github.com/chenhaoxing/DiffusionInst). Thanks very much.

**Requirements**
1. Windows, Linux or macOS with Python ≥ 3.6
2. PyTorch ≥ 1.9.0 and torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this
3. OpenCV is optional and needed by demo and visualization

**Steps**
1. Install Detectron2 following https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#installation.
2. Prepare datasets.

**Train and Test**
1. Train DiffSARShipInst
   
```
python train_net.py --num-gpus 1 --config-file configs/DiffSARShipInst-SSDD-train.yaml
```

3. Test DiffSARShipInst
   
```
python train_net.py --num-gpus 1 --config-file configs/DiffSARShipInst-SSDD-test.yaml --eval-only MODEL.WEIGHTS path/to/model.pth
```
