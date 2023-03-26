## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import pytorch_lightning as pl

from utils import img2patch


class AttentionBlock(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_heads, dropout):
        """
        emb_dim: Dimension of input and attention feature vectors
        hidden_dim: Dimension of hidden in FFN (larger 2-4x than emb_dim)
        num_heads: number of heads in Multi-head Attention
        dropout: amount of dropout in FFN
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout
        )  # , batch_first=True
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.ffn(self.layer_norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        emb_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout,
    ):
        super().__init__()

        self.patch_size = patch_size

        self.input_layer = nn.Linear(num_channels * (patch_size**2), emb_dim)
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(emb_dim, hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.mlp = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # cls_token + positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + num_patches, emb_dim))

    def forward(self, x):
        # preprocess x
        x = img2patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # add cls token and pos_emb
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_emb[:, : T + 1]

        # transformer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # predict class
        cls = x[0]
        out = self.mlp(cls)
        return out


class ViT(pl.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        if mode == "val" or mode == "test":
            self.model.eval()
            with torch.no_grad():
                preds = self.model(imgs)
            self.model.train()
        else:
            preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
