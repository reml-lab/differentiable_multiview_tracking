import cv2
import numpy as np
import torch
from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class ResizeImg(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = cv2.resize(img, self.size)
        return img

@TRANSFORMS.register_module()
class ImgToTensor(object):
    def __call__(self, img):
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        return img
