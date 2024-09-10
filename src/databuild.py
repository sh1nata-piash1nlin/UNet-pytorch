import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchmetrics
from torchmetrics import Dice, JaccardIndex
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
from glob import glob


class OxfordIIITPet(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.image_path = []
        self.txt_path = []

        if train:
            self.txt_path = os.path.join(root, "annotations", "trainval.txt")
        else:
            self.txt_path = os.path.join(root, "annotations", "test.txt")

        with open(self.txt_path) as file_in:
            for line in file_in:
                self.image_path.append(line.split(" ")[0])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", "{}.jpg".format(self.image_path[idx]))
        mask_path = os.path.join(self.root, "annotations", "trimaps", "{}.png".format(self.image_path[idx]))
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # fg=1, bg=0, nc=1
        mask = np.where(mask == 3, 1, np.where(mask == 2, 0, mask)) #binary semantic segmentation
        transformed_image = []
        transformed_mask = []
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
        return transformed_image, transformed_mask

class Visualization(object):        #how training images look like after transforming.
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            return tensor

unorm = Visualization(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

if __name__ == "__main__":
    train_size = 384
    train_transform = A.Compose([
        A.Resize(width=train_size, height=train_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(),
        A.Sharpen(),
        A.RGBShift(),
        A.Cutout(num_holes=5, max_h_size=25, max_w_size=25, fill_value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=55.0), #mean and std of ImageNet
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(width=train_size, height=train_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=55.0),
        ToTensorV2(),
    ])

    dataset = OxfordIIITPet(root="../data/OxfordIIITPet", train=True, transform=train_transform)
    img, msk = dataset.__getitem__(10)
    print(img.shape, msk.shape)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=4)
    # for images, masks in dataloader:
    #     print(images.shape)
    #     print(masks.shape)
    plt.subplot(1,2,1)
    plt.imshow(unorm(img).permute(1,2,0))
    plt.subplot(1,2,2)
    plt.imshow(msk)
    plt.show()














