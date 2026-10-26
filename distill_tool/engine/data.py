# Adapted from YOLOv5 v7.0 utils/dataloaders.py, GPL-3.0.
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.augmentations import augment_hsv, letterbox, mixup, random_perspective
from utils.dataloaders import LoadImagesAndLabels, seed_worker
from utils.general import xywhn2xyxy, xyxy2xywhn

from distill_tool.config import network_shape


def input_shape():
    return tuple(int(x) for x in os.environ.get("DISTILL_INPUT_HW", "320,328").split(","))


def pad_canvas(img, labels, external_hw):
    eh, ew = external_hw
    nh, nw = network_shape(eh, ew)
    if tuple(img.shape[:2]) != (eh, ew):
        raise ValueError(f"External image must be {external_hw}, got {img.shape[:2]}")
    labels = labels.copy()
    if len(labels):
        labels[:, [1, 3]] *= ew / nw
        labels[:, [2, 4]] *= eh / nh
    img = cv2.copyMakeBorder(img, 0, nh - eh, 0, nw - ew, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, labels


class FixedShapeDataset(LoadImagesAndLabels):
    def __init__(self, *args, **kwargs):
        self.external_hw = input_shape()
        # A fixed canvas allows shuffled training and does not require aspect buckets.
        kwargs["rect"] = False
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            img, labels = self.load_mosaic(index)
            if random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
            img, ratio, pad = letterbox(img, self.external_hw, auto=False, scaleup=True)
            if len(labels):
                labels[:, [1, 3]] = labels[:, [1, 3]] * ratio[0] + pad[0]
                labels[:, [2, 4]] = labels[:, [2, 4]] * ratio[1] + pad[1]
            shapes = None
        else:
            img = cv2.imread(self.im_files[index])
            if img is None:
                raise ValueError(f"Cannot decode image: {self.im_files[index]}")
            h0, w0 = img.shape[:2]
            img, ratio, pad = letterbox(img, self.external_hw, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((ratio[1], ratio[0]), pad)
            labels = self.labels[index].copy()
            if len(labels):
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w0, ratio[1] * h0,
                                          padw=pad[0], padh=pad[1])
            if self.augment:
                img, labels = random_perspective(img, labels, degrees=hyp["degrees"], translate=hyp["translate"],
                    scale=hyp["scale"], shear=hyp["shear"], perspective=hyp["perspective"])
        if len(labels):
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
        if self.augment:
            img, labels = self.albumentations(img, labels)
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]
        img, labels = pad_canvas(img, labels, self.external_hw)
        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes


def create_fixed_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None,
                            augment=False, cache=False, pad=0, rect=False, rank=-1,
                            workers=8, image_weights=False, quad=False, prefix="", shuffle=False):
    if rank != -1 or quad or image_weights:
        raise ValueError("Fixed-input adapter supports single GPU, standard batches and uniform sampling")
    dataset = FixedShapeDataset(path, imgsz, batch_size, augment=augment, hyp=hyp, rect=False,
        cache_images=False, single_cls=single_cls, stride=int(stride), pad=0, prefix=prefix)
    generator = torch.Generator().manual_seed(int(os.environ.get("DISTILL_SEED", "42")))
    batch_size = min(batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=min(workers, os.cpu_count() or 1),
        pin_memory=True, collate_fn=LoadImagesAndLabels.collate_fn, worker_init_fn=seed_worker, generator=generator)
    return loader, dataset
