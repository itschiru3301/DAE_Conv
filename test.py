import os
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from custom_dataset import MyDataset
from ConvDAE import Autoencoder as DenoiseAE64


@torch.no_grad()
def main():
    test_root = "/data/b23_chiranjeevi/CAE_Denoiser/dataset/test"   # contains noise/ and label/
    ckpt_path = "dae64.pt"
    out_dir = "test_outputs"
    bs = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    ds = MyDataset(test_root, transforms=tfm)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    model = DenoiseAE64(out_activation="sigmoid").to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    # Save a grid preview from the first batch: noisy | denoised | clean
    noisy, clean = next(iter(dl))
    noisy = noisy.to(device)
    clean = clean.to(device)
    den = model(noisy).clamp(0, 1)

    n = min(8, noisy.size(0))
    grid = make_grid(torch.cat([noisy[:n].cpu(), den[:n].cpu(), clean[:n].cpu()], dim=0), nrow=n)
    save_image(grid, os.path.join(out_dir, "preview_grid.png"))

    # Save all denoised images individually
    idx = 0
    for noisy, clean in dl:
        noisy = noisy.to(device)
        den = model(noisy).clamp(0, 1).cpu()  # (B,3,64,64)

        for b in range(den.size(0)):
            save_image(den[b], os.path.join(out_dir, f"den_{idx:06d}.png"))
            idx += 1

    print(f"Saved {idx} denoised images to {out_dir}/")


if __name__ == "__main__":
    main()
