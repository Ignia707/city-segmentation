import torch
import numpy as np


class MeanIoU:
    """
    Computes mean Intersection over Union (mIoU) for semantic segmentation.
    Ignores label 255 (Cityscapes void/unlabeled class).
    """
    def __init__(self, num_classes=19, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset confusion matrix at start of each epoch."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64
        )

    def update(self, preds, labels):
        """
        preds  : torch.Tensor (B, H, W) — predicted class per pixel
        labels : torch.Tensor (B, H, W) — ground truth class per pixel
        """
        preds = preds.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()

        # remove ignore index pixels
        valid = labels != self.ignore_index
        preds = preds[valid]
        labels = labels[valid]

        # accumulate into confusion matrix
        np.add.at(
            self.confusion_matrix,
            (labels, preds),
            1
        )

    def compute(self):
        """Compute mIoU from accumulated confusion matrix."""
        cm = self.confusion_matrix
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection

        # avoid division by zero for classes not present
        iou_per_class = np.where(
            union > 0,
            intersection / union,
            np.nan
        )

        # mean over classes that actually appear
        miou = np.nanmean(iou_per_class)
        return miou, iou_per_class