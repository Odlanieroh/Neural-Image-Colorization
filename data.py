import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import datasets
from skimage.color import rgb2lab, lab2rgb

import config


def lab_to_rgb(L, ab):
    # The network predicts normalized Lab chrominance, so restore its native range before RGB conversion.
    lab = np.concatenate([L, ab * 128.0], axis=-1).astype("float64")
    return np.clip(lab2rgb(lab), 0, 1)


class LabDataset(Dataset):
    def __init__(self, base, augment):
        self.base = base
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img = self.base[i][0].convert("RGB")
        if self.augment:
            w, h = img.size
            s = random.uniform(0.8, 1.0)
            cw, ch = max(1, int(w * s)), max(1, int(h * s))
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            img = img.crop((x, y, x + cw, y + ch))
        img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.rotate(random.uniform(-15, 15))
        # Separate luminance from color so grayscale L is the input and normalized a/b channels are the target.
        lab = rgb2lab(np.asarray(img, dtype="float32") / 255.0).astype("float32")
        L = torch.from_numpy(lab[:, :, :1]).permute(2, 0, 1)
        ab = torch.from_numpy(lab[:, :, 1:] / 128.0).permute(2, 0, 1)
        return L, ab


def _build_base():
    sources = [
        ("Flowers102 (plants)",
         lambda: [datasets.Flowers102(root="data", split=s, download=True)
                  for s in ("train", "val", "test")]),
        ("OxfordIIITPet (cats/dogs)",
         lambda: [datasets.OxfordIIITPet(root="data", split=s, download=True)
                  for s in ("trainval", "test")]),
        ("STL10 (vehicles/animals)",
         lambda: [datasets.STL10(root="data", split=s, download=True)
                  for s in ("train", "test")]),
        ("VOC 2012 (people + scenes)",
         lambda: [datasets.VOCDetection(root="data", year="2012",
                                        image_set="trainval", download=True)]),
    ]
    parts = []
    for name, builder in sources:
        try:
            parts += builder()
            print(f"  [data] loaded {name}")
        except Exception as e:
            print(f"  [data] SKIPPED {name} -> {type(e).__name__}: {e}")
    if not parts:
        raise RuntimeError("No datasets could be downloaded -- check your network.")
    return ConcatDataset(parts)


def load_dataloaders():
    base = _build_base()
    n_val = 300
    # A fixed seed keeps the 300-image validation set reproducible across training and evaluation.
    train_raw, val_raw = random_split(
        base, [len(base) - n_val, n_val],
        generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(LabDataset(train_raw, True),
                              batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(LabDataset(val_raw, False),
                            batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader
