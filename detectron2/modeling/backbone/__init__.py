# Copyright (c) Facebook, Inc. and its affiliates.
from .build import build_backbone, BACKBONE_REGISTRY  # noqa F401 isort:skip

from .backbone import Backbone
from .fpn import FPN, FPN_1p_5p, SCJE_FPN, FPN_1p_4p
from .regnet import RegNet
from .resnet import (
    BasicStem,
    ResNet,
    ResNetBlockBase,
    build_resnet_backbone,
    make_stage,
    BottleneckBlock,
)
from .vit import ViT, SimpleFeaturePyramid, get_vit_lr_decay_rate
from .mvit import MViT
from .swin import SwinTransformer
from .fan import FAN
from detectron2_backbone.backbone.bifpn import build_resnet_bifpn_p1_p5_backbone
# from .dcn_v2 import DCN

__all__ = [k for k in globals().keys() if not k.startswith("_")]
# TODO can expose more resnet blocks after careful consideration
