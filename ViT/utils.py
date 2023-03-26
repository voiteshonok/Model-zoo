from torchvision import transforms


def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.49139968, 0.48215841, 0.44653091],
                [0.24703223, 0.24348513, 0.26158784],
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.49139968, 0.48215841, 0.44653091],
                [0.24703223, 0.24348513, 0.26158784],
            ),
        ]
    )

    return train_transform, test_transform


def img2patch(x, patch_size, flatten_channels=True):
    """
    x: torch.Tensor - batch of images [B, C, H, W]
    patch_size: integer - number of patches
    flatten_channels: bool - flatten format if True else image grid
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_h, p_W]
    x = x.flatten(1, 2)
    return x.flatten(2, 4) if flatten_channels else x # [B, H'*W', C, p_H, p_W] or [B, H'*W', C*p_H*p_W]
