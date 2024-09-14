import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.databuild import OxfordIIITPet
from src.unet import Unet
from src.utils import *
import torchmetrics
from torchmetrics import Dice, JaccardIndex
import segmentation_models_pytorch as smp
from tqdm import tqdm
from glob import glob
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import shutil
import numpy as np




def train(args):
    torch.cuda.empty_cache()
    train_transform = A.Compose([
        A.Resize(width=args.image_size, height=args.image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(),
        A.Sharpen(),
        A.RGBShift(),
        A.Cutout(num_holes=5, max_h_size=25, max_w_size=25, fill_value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=55.0),
        # mean and std of ImageNet
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(width=args.image_size, height=args.image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=55.0),
        ToTensorV2(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = OxfordIIITPet(root=args.data_path, train=True, transform=train_transform)
    valid_set = OxfordIIITPet(root=args.data_path, train=False, transform=test_transform)



    training_params ={
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "drop_last": True,
        "shuffle": True,
    }

    valid_params = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "drop_last": False,
        "shuffle": False,
    }

    training_dataloader = DataLoader(train_set, **training_params)
    valid_dataloader = DataLoader(valid_set, **training_params)

    model = Unet(1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = None
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    dice_fn = torchmetrics.Dice(num_classes=2, average='macro').to(device) #for pet and bg
    iou_fn = torchmetrics.JaccardIndex(num_classes=2, task='binary', average='macro').to(device)
    acc_meter = AverageMeter()
    train_loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()


    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
    else:
        start_epoch = 0
        best_acc = 0

    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.tensorboard_path)

    total_iters = len(training_dataloader)
    for epoch in range(start_epoch, args.epochs):
        acc_meter.reset()
        train_loss_meter.reset()
        dice_meter.reset()
        iou_meter.reset()
        model.train()
        progress_bar = tqdm(training_dataloader, colour='cyan')
        for iter, (images, labels) in enumerate(progress_bar):
            print(type(images), type(labels))
            n = images.shape[0]
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            prediction = prediction.squeeze() #logit (-8, 8)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()

            prediction_mask = prediction.sigmoid().round() #mask: 0 & 1
            dice_score = dice_fn(prediction, labels)
            iou_score = iou_fn(prediction, labels)
            accuracy = accuracy_fn(prediction, labels)
            train_loss_meter.update(loss.item(), n)
            iou_meter.update(iou_score.item(), n)
            dice_meter.update(dice_score.item(), n)
            acc_meter.update(accuracy.item(), n)
            progress_bar.set_description("EP {}/{}, train_loss = {}, acc = {}, IoU = {}, dice = {}".format(epoch+1, args.epochs,
                train_loss_meter, acc_meter, iou_meter, dice_meter))
            writer.add_scalar("Train", train_loss_meter.avg, acc_meter.avg, iou_meter.avg, dice_meter.avg, epoch*total_iters+iter)

        #Valid

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "batch_size": args.batch_size,
        }

        torch.save(checkpoint, os.path.join(args.trained_models, "last.pth"))
        if acc_meter.avg > best_acc:
            torch.save(checkpoint, os.path.join(args.trained_models, "best.pth"))
            best_acc = acc_meter



if __name__ == '__main__':
    args = get_args()
    train(args)





















