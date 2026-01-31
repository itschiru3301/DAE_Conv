# train.py  (simple)
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from custom_dataset import MyDataset
from ConvDAE import Autoencoder as DenoiseAE64


# -------- Xavier init --------
def init_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@torch.no_grad()
def save_samples(model, loader, device, out_path, n=8):
    model.eval()
    noisy, clean = next(iter(loader))
    noisy = noisy[:n].to(device)
    clean = clean[:n].to(device)

    den = model(noisy).clamp(0, 1)

    # rows: noisy | denoised | clean
    viz = torch.cat([noisy.cpu(), den.cpu(), clean.cpu()], dim=0)
    grid = make_grid(viz, nrow=n)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)
    model.train()


def main():
    root = "/data/b23_chiranjeevi/CAE_Denoiser/dataset/train"   # contains noise/ and label/
    epochs = 30
    bs = 16
    lr = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 224x224 -> 64x64, then ToTensor() gives [0,1]
    tfm = transforms.Compose([
        transforms.ToPILImage(),          # because Dataset returns numpy RGB
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    ds = MyDataset(root, transforms=tfm)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)

    model = DenoiseAE64(out_activation="sigmoid").to(device)
    model.apply(init_xavier)  # apply Xavier init

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        running = 0.0

        for noisy, clean in dl:
            noisy = noisy.to(device)
            clean = clean.to(device)

            pred = model(noisy)
            loss = criterion(pred, clean)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += loss.item()

        print(f"epoch {ep+1}/{epochs}  loss: {running/len(dl):.4f}")

        # save preview each epoch
        save_samples(model, dl, device, f"outputs/epoch_{ep+1:03d}.png", n=8)

    torch.save(model.state_dict(), "dae64.pt")
    print("saved dae64.pt")


if __name__ == "__main__":
    main()
