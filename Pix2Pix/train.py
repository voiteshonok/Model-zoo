import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, display_progress
import torch.nn as nn
import torch.optim as optim
import config
from dataset import FacadesDataset
from generator import UnetGenerator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image


def train(D, G, train_loader, opt_D, opt_G, L1_loss, bce_loss):
    loop = tqdm(train_loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        y_fake = G(x)
        D_real = D(x, y)
        D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        D_fake = D(x, y_fake.detach())
        D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # Train generator
        D_fake = D(x, y_fake)
        G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
        L1 = L1_loss(y_fake, y) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    D = Discriminator(in_channels=3).to(config.DEVICE)
    G = UnetGenerator(in_channels=3, features=64).to(config.DEVICE)
    opt_D = optim.Adam(D.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE_loss = nn.CrossEntropyLoss()
    L1_loss = nn.L1Loss()

    train_dataset = FacadesDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_dataset = FacadesDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train(D, G, train_loader, opt_D, opt_G, L1_loss, BCE_loss)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(G, opt_G, filename=config.CHECKPOINT_GEN)
            save_checkpoint(D, opt_D, filename=config.CHECKPOINT_DISC)

            G.eval()
            for x, y in val_loader:
                y_fake = G.generate(x.to(config.DEVICE))
                display_progress(x, y_fake, y)
                break
            G.train()

        save_some_examples(G, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()
