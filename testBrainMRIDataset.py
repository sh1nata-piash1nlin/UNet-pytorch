import argparse
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from src.unetPlus import UNetPlusPlus

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("-t", "--test_folder", type=str, default="../doctor_image",
                   help="Directory with test images")
    p.add_argument("-c", "--checkpoint_path", type=str, default="trained_model/bestBrainMRI.pth")
    p.add_argument("-s", "--image_size", type=int, default=256)
    p.add_argument("-b", "--batch_size", type=int, default=4)
    return p.parse_args()

def test_and_viz(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = UNetPlusPlus().to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # gather a small batch of PIL→Tensor
    img_fnames = sorted(os.listdir(args.test_folder))[: args.batch_size]
    imgs = []
    for fn in img_fnames:
        pil = Image.open(os.path.join(args.test_folder, fn)).convert("RGB")
        imgs.append(preprocess(pil))
    batch = torch.stack(imgs, dim=0).to(device)          # [B,3,H,W]

    # forward + sigmoid + threshold
    with torch.no_grad():
        logits = model(batch)                             # [B,1,H,W]
        probs  = torch.sigmoid(logits)                    # [B,1,H,W]
        preds  = (probs > 0.5).float()                    # [B,1,H,W]

    # move back to CPU for plotting
    batch_np = batch.cpu().permute(0,2,3,1).numpy()      # [B,H,W,3]
    preds_np = preds.cpu().squeeze(1).numpy()            # [B,H,W]

    # plot
    B = batch_np.shape[0]
    fig, axes = plt.subplots(B, 2, figsize=(6, 3*B))
    if B == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(B):
        axes[i,0].imshow(batch_np[i])
        axes[i,0].set_title("Input")
        axes[i,0].axis("off")

        axes[i,1].imshow(preds_np[i], cmap="gray")
        axes[i,1].set_title("Predicted mask")
        axes[i,1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = get_args()
    test_and_viz(args)
