from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2
import os

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

from DiffSARShipInst import add_DiffSARShipInst_config
from DiffSARShipInst.util.model_ema import add_model_ema_configs

def Predict():
    # register_coco_instances("coco_2017_val", {}, "datasets/coco/annotations/instances_train2017.json", "datasets/coco/train2017")
    custom_metadata = MetadataCatalog.get("coco_2017_val")
    DatasetCatalog.get("coco_2017_val")

    im = cv2.imread("datasets/coco/images/test/000001.jpg")
    cfg = get_cfg()
    # add_DiffSARShipInst_config(cfg)
    # add_model_ema_configs(cfg)
    cfg.merge_from_file(
        # "F:\DiffSARShipInst\configs\diffinst.ssdd.res50.inst.yaml"
    "F:\detectron2\configs\COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_1x.yaml"
    )
    cfg.DATASETS.TEST = ("coco_2017_val", )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "F:\DiffSARShipInst\model\model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        512        ###128
    )  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=custom_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('Result',v.get_image()[:, :, ::-1])
    cv2.waitKey()


if __name__ == "__main__":
    Predict()
