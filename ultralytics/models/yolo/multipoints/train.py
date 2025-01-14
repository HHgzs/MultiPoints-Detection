# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn

from ultralytics.models import yolo
from ultralytics.nn.tasks import MultiPointsModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.data import build_dataloader, build_yolo_dataset


class MultiPointsTrainer(yolo.detect.DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a MultiPointsTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "multipoints"
        super().__init__(cfg, overrides, _callbacks)
        

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return MultiPointsModel initialized with specified config and weights."""
        model = MultiPointsModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.multipoints.MultiPointsValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch
    
    def build_dataset(self, img_path, mode="train", batch=None):
        m = self.model.model[-1]  # last layer
        self.nc = m.nc    # number of classes
        self.n_p = m.n_p  # number of points
        
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, n_p=self.n_p)


    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            multipoints=batch["multipoints"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, multipoints=True, on_plot=self.on_plot)  # save results.png
