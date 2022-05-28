import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                bias=False,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(
                out_channels, affine=True
            ),  # better than BatchNorm2d with artefacts
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):  # 256 -> 30x30
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(
                    in_channels, feature, stride=1 if feature == features[-1] else 2
                )
            )
            in_channels = feature

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)

        x = self.pooling(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)

        return x


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x)
    print(preds.shape)
    # assert preds.shape == torch.Size([1, 1, 30, 30])


test()
