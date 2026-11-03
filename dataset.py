from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


def read_xray(file_path):
    xray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) # H W
    if xray is None:
        raise FileNotFoundError(file_path)

    xray = xray.astype(np.float32) / 255.0
    xray = xray.reshape(1, *xray.shape) # 1 H W

    return xray


def read_mask(file_path):
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(file_path)
    return mask


class knee_segmentation_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        xray = read_xray(str(ROOT / self.dataset["xrays"].iloc[idx]))
        mask = read_mask(str(ROOT / self.dataset["masks"].iloc[idx]))

        xray = torch.from_numpy(xray)
        # Masks are stored as 0/255.
        mask = torch.from_numpy(mask).unsqueeze(0).float().div(255)

        return {"xray": xray, "mask": mask}
