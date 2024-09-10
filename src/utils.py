import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def get_args():
    parse = argparse.ArgumentParser(description='Animal')
    parse.add_argument('-p', '--data_path', type=str, default='./data/OxfordIIITPet/')
    parse.add_argument('-b', '--batch_size', type=int, default=32)
    parse.add_argument('-w', '--num_workers', type=int, default= os.cpu_count()  )
    parse.add_argument('-e', '--epochs', type=int, default=100)
    parse.add_argument("-p", "--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parse.add_argument('-l', '--lr', type=float, default=1e-3) #for adam
    parse.add_argument('-s', '--image_size', type=int, default=224)
    parse.add_argument('-c', '--checkpoint_path', type=str, default=None) #None = train tu dau
    parse.add_argument('-t', '--tensorboard_path', type=str, default="tensorboard")
    parse.add_argument('-r', '--trained_models', type=str, default="trained_models")
    args = parse.parse_args()
    return args

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_fn(preds, targets):
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    acc = torch.sun(preds_flat == targets_flat)
    return acc/targets_flat.shape[0]

