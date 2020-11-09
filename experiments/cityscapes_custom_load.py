import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np

from PIL import Image

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager, file_lock


from detectron2.data import MetadataCatalog, DatasetCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_cityscapes_train",]


def load_cityscapes_train(temp_scaling = False):

	## when temprature scaling is being learned, we load a different split
	if temp_scaling:
		print("Learning temprature scaling parameter!")
		return list(np.load('cityscapes_temp_finetune_split.npy', allow_pickle=True))
	else:
		print("Learning model for temprature scaling!")
		return list(np.load('cityscapes_train_split.npy', allow_pickle=True))
