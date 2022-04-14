import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import config


class FacadesDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :256, :]
        target_image = image[:, 256:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = FacadesDataset("facades/train")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        break
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    for i in range(1):
        ax[i][0].imshow(dataset.__getitem__(i)[0].permute(1, 2, 0))
        ax[i][1].imshow(dataset.__getitem__(i)[1].permute(1, 2, 0))
