import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, display_progress
import torch.nn as nn
import torch.optim as optim
import config
from dataset import FacadesDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image


def train_fn(disc_M, disc_R, gen_M, gen_R, loader, opt_disc, opt_gen, l1, mse):
    R_reals = 0
    R_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (mask, real) in enumerate(loop):
        mask = mask.to(config.DEVICE)
        real = real.to(config.DEVICE)

        # Train Discriminators
        fake_real = gen_R(mask)
        D_R_real = disc_R(real)
        D_R_fake = disc_R(fake_real.detach())
        R_reals += D_R_real.mean().item()
        R_fakes += D_R_fake.mean().item()
        D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
        D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
        D_R_loss = D_R_real_loss + D_R_fake_loss

        fake_mask = gen_M(real)
        D_M_real = disc_M(mask)
        D_M_fake = disc_M(fake_mask.detach())
        D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
        D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
        D_M_loss = D_M_real_loss + D_M_fake_loss

        # put it togethor
        D_loss = (D_R_loss + D_M_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train Generators
        # adversarial loss for both generators
        D_R_fake = disc_R(fake_real)
        D_M_fake = disc_M(fake_mask)
        loss_G_R = mse(D_R_fake, torch.ones_like(D_R_fake))
        loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))

        # cycle loss
        cycle_mask = gen_M(fake_real)
        cycle_real = gen_R(fake_mask)
        cycle_mask_loss = l1(mask, cycle_mask)
        cycle_real_loss = l1(real, cycle_real)

        # add all togethor
        G_loss = (
            loss_G_R
            + loss_G_M
            + cycle_mask_loss * config.LAMBDA_CYCLE
            + cycle_real_loss * config.LAMBDA_CYCLE
        )

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if idx % 5 == 0:
            save_image(fake_real * 0.5 + 0.5, f"saved_images/real_{idx}.png")
            save_image(fake_mask * 0.5 + 0.5, f"saved_images/mask_{idx}.png")

        loop.set_postfix(H_real=R_reals / (idx + 1), H_fake=R_fakes / (idx + 1))


def main():
    disc_M = Discriminator(in_channels=3).to(config.DEVICE)  # for mask
    disc_R = Discriminator(in_channels=3).to(config.DEVICE)  # for real
    gen_M = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_R = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_M.parameters()) + list(disc_R.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_M.parameters()) + list(gen_R.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    train_dataset = FacadesDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    # val_dataset = FacadesDataset(root_dir=config.VAL_DIR)
    # val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_M, disc_R, gen_M, gen_R, train_loader, opt_disc, opt_gen, L1, mse)

        if config.SAVE_MODEL and (epoch + 1) % 5 == 0:
            save_checkpoint(gen_M, opt_gen, filename="gen_M.pth")
            save_checkpoint(gen_R, opt_gen, filename="gen_R.pth")
            save_checkpoint(disc_R, opt_disc, filename="disc_R.pth")
            save_checkpoint(disc_M, opt_disc, filename="disc_M.pth")


if __name__ == "__main__":
    main()
