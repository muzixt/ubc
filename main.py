import sys

import numpy as np
import pandas as pd
from pathlib import Path
import os
import torch
from PIL import Image
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNeXt50_32X4D_Weights
from torch import nn
import torchvision.models as models
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import DualTransform
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder, EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

root = Path('/home/cqwu/lw/UBC')
train_dir = root / "train_crops"


class CancerThumbnailDataset(Dataset):

    def __init__(self, img_root_dir, transforms=None, mode: str = 'train', split: float = 0.90):
        self.split = split
        self.img_root_dir = img_root_dir
        self.transforms = None
        self.mode = mode
        self.labels_dic = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
        self.transforms = transforms
        self.imgs = []
        self.labels = []
        for la in self.labels_dic:
            ims = self._get_files(self.img_root_dir / la)
            frac = int(self.split * len(ims))
            imm = ims[:frac] if mode == 'train' else ims[frac:]

            self.imgs += imm
            self.labels += [la] * len(imm)

    def _get_files(self, path: Path):
        return list(path.glob("*"))

    def __getitem__(self, idx: int) -> tuple:
        img = cv.imread(str(self.imgs[idx]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.mode == "test":
            if self.transforms:
                img = self.transforms(image=img)['image']
            return img
        else:
            label = self.labels_dic.index(self.labels[idx])

            if self.transforms:
                img = self.transforms(image=img)['image']

            return img, label

    def __len__(self) -> int:
        return len(self.imgs)


class Net(pl.LightningModule):
    def __init__(self, num_classes=5, lr=1e-3):
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, 500)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(500, 100)
        self.fc1 = nn.Linear(100, num_classes)
        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.fc1(x)
        # x = self.sf(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).sum().item() / len(preds)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': self.log}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).sum().item() / len(preds)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, "val_acc": acc, "log": self.log}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_acc"}

    def configure_callbacks(self):
        model_checkpoint = ModelCheckpoint(
            monitor='val_acc',
            filename='best-model-{epoch:02d}-{val_acc:.2f}',
            save_top_k=1,
            save_last=True,
            mode='max',
            save_on_train_epoch_end=False
        )
        lr_finder = LearningRateFinder()
        early_stopping = EarlyStopping(
            monitor='val_acc',
            patience=20,
            mode='max',
            check_on_train_epoch_end=False
        )
        return [model_checkpoint, lr_finder]


def hisEqul(img, k=7, clip_limit=1.0):
    ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCR_CB)
    channels = cv.split(ycrcb)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(k, k))
    clahe.apply(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2RGB, img)
    return img


class HisEqul(DualTransform):
    def __init__(self, k=7, clip_limit=1.0, always_apply: bool = False, p: float = 0.5):
        super(HisEqul, self).__init__(always_apply, p)
        self.k = k
        self.clip_limit = clip_limit

    def apply(self, img, **params) -> np.ndarray:
        return hisEqul(img, self.k, self.clip_limit)


train_transforms = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.25, rotate_limit=180, p=.2),
    # A.RandomCrop(256, 256, p=0.2),
    A.RandomGridShuffle(p=0.2),
    A.Resize(224, 224),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8),
    A.VerticalFlip(),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
    A.GaussianBlur(blur_limit=(5, 9), sigma_limit=(0.1, 5), p=.5),
    HisEqul(k=7, clip_limit=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225], ),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225], ),
    ToTensorV2(),
])

bs = 64
num_workers = 8

train_dataset = CancerThumbnailDataset(img_root_dir=train_dir, transforms=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, persistent_workers=True)
val_dataset = CancerThumbnailDataset(img_root_dir=train_dir, transforms=val_transforms, mode="val")
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=num_workers, persistent_workers=True)

model = Net(lr=0.0001)

logger = CSVLogger("logs", name="ubc")
trainer = pl.Trainer(max_epochs=500, accelerator='gpu' if torch.cuda.is_available() else 'cpu', log_every_n_steps=50,
                     logger=logger, check_val_every_n_epoch=1)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
