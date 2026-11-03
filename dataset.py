from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


def read_img(file_path):
    xray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if xray is None:
        raise FileNotFoundError(file_path)
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
        img = read_img(str(ROOT / self.dataset["xrays"].iloc[idx]))
        mask = read_mask(str(ROOT / self.dataset["masks"].iloc[idx]))

        # Grayscale to normalized 3-channel tensor for the ImageNet backbone.
        xray = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).float().div(255)
        # Masks are stored as 0/255.
        mask = torch.from_numpy(mask).unsqueeze(0).float().div(255)

        return {"xray": xray, "mask": mask}
