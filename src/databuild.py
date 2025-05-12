import numpy as np
import cv2
import torch
from torch.utils import data
import os

HERE = os.path.dirname(__file__)
root = os.path.join(HERE, "..", "data", "lgg-mri-segmentation", "kaggle_3m")

no_mask = 0
no_mask_files = []
no_file = 0
no_files = []
num_empty_masks = 0
num_nonempty_masks = 0

S = 3  # number of pos/neg samples to take
empty_mask_samples = []
nonempty_mask_samples = []

img_dimensions = []
msk_dimensions = []

n_files = 0
for directory in [os.path.join(root, x) for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]:
    for file in os.listdir(directory):
        n_files += 1
        img_dimensions.append(np.array(cv2.imread(os.path.join(directory, file))).shape)
        # count files with no mask
        if 'mask' not in file:
            # check if mask exists
            mask_path = os.path.join(directory, file[:file.find('.tif')] + '_mask.tif')
            if not os.path.exists(mask_path):
                no_mask += 1
                no_mask_files.append(os.path.join(directory, file))
        else:
            msk_dimensions.append(np.array(cv2.imread(os.path.join(directory, file), cv2.IMREAD_UNCHANGED)).shape)
            # count masks with no file
            f_path = os.path.join(directory, file[:file.find('mask') - 1] + '.tif')
            # check if file exists
            if not os.path.exists(f_path):
                no_file += 1
                no_files.append(os.path.join(directory, file))

            # check if mask is empty
            j = np.max(cv2.imread(os.path.join(directory, file), cv2.IMREAD_UNCHANGED))
            if j > 0:
                num_nonempty_masks += 1
                if len(nonempty_mask_samples) < S:
                    nonempty_mask_samples.append(os.path.join(directory, file))
            else:
                num_empty_masks += 1
                if len(empty_mask_samples) < S:
                    empty_mask_samples.append(os.path.join(directory, file))

file_list = []
for directory in [os.path.join(root, x) for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]:
    for file in os.listdir(directory):
        # add files to list
        if 'mask' not in file:
            result = 0
            img_path = os.path.join(directory, file)
            mask_path = os.path.join(directory, file[:file.find('.tif')] + '_mask.tif')

            # check if mask is nonempty
            if np.max(cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)) > 0:
                result = 1

            file_list.append([img_path, mask_path, result])


class Brain_MRI_Segmentation_Dataset(data.Dataset):
    def __init__(self, inputs, transform=None):
        self.inputs = inputs
        self.transform = transform
        self.input_dtype = torch.float32
        self.target_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # for classification return only the image and the binary label
        img_path = self.inputs[index][0]
        mask_path = self.inputs[index][1]
        # mask_img = cv2.normalize(cv2.imread(mask_path), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        x = torch.from_numpy(np.transpose(np.array(cv2.imread(img_path)), (2, 0, 1))).type(self.input_dtype)
        y = torch.from_numpy(np.resize(np.array(mask_img) / 255., (1, 256, 256))).type(self.target_dtype)

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        return x, y
