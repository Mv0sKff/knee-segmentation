import cv2
import torch
from torch.utils.data import Dataset


def read_img(file_path):
    xray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return xray

def read_mask(file_path):
    return cv2.imread(file_path)

class knee_segmentation_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = read_img(self.dataset["xrays"].iloc[idx])
        mask = read_mask(self.dataset["masks"].iloc[idx])

        # work around: Grayscale to normalized 3-channel tensor.
        #img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).float().div(255)

        result = {"xray": img, "mask": mask}

        return result
