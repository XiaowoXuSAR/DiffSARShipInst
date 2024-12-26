"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

Modified by Xiaowo Xu
Date: Dec 25, 2024
Contact: xuxiaowo@std.uestc.edu.cn
"""

from detectron2.config import CfgNode as CN

def add_DiffSARShipInst_config(cfg):
    """
    Add config for DiffSARShipInst_ResNet50
    """
    cfg.MODEL.DiffSARShipInst = CN()
    cfg.MODEL.DiffSARShipInst.NUM_CLASSES = 80
    cfg.MODEL.DiffSARShipInst.NUM_PROPOSALS = 300

    ################################################
    # SCJE_FPN_Model
    cfg.MODEL.SCJE_FPN = CN()
    cfg.MODEL.SCJE_FPN.FUSE_TYPE = 'sum'
    cfg.MODEL.SCJE_FPN.IN_FEATURES = ('res2', 'res3', 'res4', 'res5')
    cfg.MODEL.SCJE_FPN.NORM = ' '
    cfg.MODEL.SCJE_FPN.OUT_CHANNELS = 256
    ################################################

    # RCNN Head.
    cfg.MODEL.DiffSARShipInst.NHEADS = 8
    cfg.MODEL.DiffSARShipInst.DROPOUT = 0.0
    cfg.MODEL.DiffSARShipInst.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffSARShipInst.ACTIVATION = 'relu'
    cfg.MODEL.DiffSARShipInst.HIDDEN_DIM = 256
    cfg.MODEL.DiffSARShipInst.NUM_CLS = 1
    cfg.MODEL.DiffSARShipInst.NUM_REG = 3
    cfg.MODEL.DiffSARShipInst.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DiffSARShipInst.NUM_DYNAMIC = 2
    cfg.MODEL.DiffSARShipInst.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffSARShipInst.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffSARShipInst.FIOU_WEIGHT = 2.0
    cfg.MODEL.DiffSARShipInst.L1_WEIGHT = 5.0
    cfg.MODEL.DiffSARShipInst.DEEP_SUPERVISION = True
    cfg.MODEL.DiffSARShipInst.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffSARShipInst.USE_FOCAL = True
    cfg.MODEL.DiffSARShipInst.USE_FED_LOSS = False
    cfg.MODEL.DiffSARShipInst.ALPHA = 0.25
    cfg.MODEL.DiffSARShipInst.GAMMA = 2.0
    cfg.MODEL.DiffSARShipInst.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffSARShipInst.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffSARShipInst.SNR_SCALE = 2.0
    cfg.MODEL.DiffSARShipInst.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffSARShipInst.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
