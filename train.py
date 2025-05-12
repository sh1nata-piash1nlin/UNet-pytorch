import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from srrc.unet import *
from srrc.unet import *
from srrc.utils import *
from torch.utils import data
import torchmetrics
from torchmetrics import Dice, JaccardIndex
import segmentation_models_pytorch as smp
from tqdm import tqdm
from glob import glob
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from srrc.databuild import Brain_MRI_Segmentation_Dataset
import cv2


def run_experiment(
    model_name,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device="cuda",
    num_epochs=50,
    ckpt_folder="trained_models",
    resume=True,
):
    os.makedirs(ckpt_folder, exist_ok=True)
    last_ckpt = os.path.join(ckpt_folder, f"{model_name}_last.pth")
    best_ckpt = os.path.join(ckpt_folder, f"{model_name}_best.pth")

    model.to(device)

    # optionally resume
    start_epoch = 0
    best_iou = -1.0
    if resume and os.path.isfile(last_ckpt):
        print(f"⏩ Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", best_iou)
        print(f"   Resumed at epoch {start_epoch}, best_iou={best_iou:.4f}")


    for epoch in range(start_epoch, num_epochs):
        # ——— TRAIN ———
        model.train()
        for imgs, masks in tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()

        # ——— VALIDATE ———
        model.eval()
        val_ious = []
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Val   {epoch+1}/{num_epochs}"):
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                probs = torch.sigmoid(out)
                val_ious.append(iou_pytorch(probs, masks).item())

        epoch_iou = mean(val_ious)
        print(f"Epoch {epoch+1}/{num_epochs} — Val mIoU: {epoch_iou:.4f}")

        # save “last” checkpoint
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "best_iou":   best_iou,
        }, last_ckpt)

        # save “best” checkpoint
        if epoch_iou > best_iou:
            best_iou = epoch_iou
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_iou":   best_iou,
            }, best_ckpt)
            print(f"🎉 New best mIoU! Saved to {best_ckpt}")

    return {
        "last_checkpoint": last_ckpt,
        "best_checkpoint": best_ckpt,
    }


def visualize_segmentation2(model, data_loader, num_samples=5, device='cuda'):
    # visualize segmentation on unseen samples
    fig, axs = plt.subplots(num_samples, 3, figsize=(60, 60))

    for ax, col in zip(axs[0], ['MRI', 'Ground Truth', 'Predicted Mask']):
        ax.set_title(col)

    index = 0
    for i, batch in enumerate(data_loader):
        img = batch[0].to(device)
        msk = batch[1].to(device)

        output = model(img)

        for j in range(batch[0].size()[0]):  # iterate over batchsize
            axs[index, 0].imshow(np.transpose(img[j].detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8), cmap='bone',
                                 interpolation='none')

            axs[index, 1].imshow(np.transpose(img[j].detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8), cmap='bone',
                                 interpolation='none')
            axs[index, 1].imshow(torch.squeeze(msk[j]).detach().cpu().numpy(), cmap='Blues', interpolation='none',
                                 alpha=0.5)

            axs[index, 2].imshow(np.transpose(img[j].detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8), cmap='bone',
                                 interpolation='none')
            axs[index, 2].imshow(torch.squeeze(output[j]).detach().cpu().numpy(), cmap='Greens', interpolation='none',
                                 alpha=0.5)

            index += 1

        if index >= num_samples:
            break

    plt.tight_layout()

if __name__ == "__main__":
    # build file list
    root = "./data/lgg-mri-segmentation/kaggle_3m"
    file_list = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if "mask" not in f:
                img_p  = os.path.join(d, f)
                msk_p  = os.path.join(d, f.replace(".tif", "_mask.tif"))
                if os.path.exists(msk_p):
                    file_list.append([img_p, msk_p])

    # only non-empty masks
    positive = [x for x in file_list
                if np.max(cv2.imread(x[1], cv2.IMREAD_UNCHANGED)) > 0]

    # your augment transform (applied to both x and y as before)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 0.8))
    ])

    # dataset & split
    full_ds = Brain_MRI_Segmentation_Dataset(positive, transform=transform)
    val_sz = int(0.3 * len(full_ds))
    train_ds, val_ds = data.random_split(full_ds, [len(full_ds)-val_sz, val_sz])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=4)

    # model, optimizer, loss
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = UNet()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    # run (with checkpoint/resume)
    ckpts = run_experiment(
        "Unet_SGD_AugmentedData",
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device=device,
        num_epochs=50,
        ckpt_folder="trained_models",
        resume=True
    )

    print("Training complete.")
    print(" Last:", ckpts["last_checkpoint"])
    print(" Best:", ckpts["best_checkpoint"])















