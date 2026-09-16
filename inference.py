import os
import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab

import config
from data import lab_to_rgb
from model import Colorizer


def load_model():
    if not os.path.exists(config.WEIGHTS_PATH):
        raise FileNotFoundError("No trained weights found - click 'Train model' first.")
    model = Colorizer().to(config.DEVICE)
    model.load_state_dict(torch.load(config.WEIGHTS_PATH, map_location=config.DEVICE))
    model.eval()
    return model


def colorize_array(model, path):
    gray = Image.open(path).convert("L").resize((config.IMG_SIZE, config.IMG_SIZE))
    g = np.asarray(gray, dtype="float32") / 255.0
    L = rgb2lab(np.stack([g, g, g], axis=-1)).astype("float32")[:, :, :1]
    tensor = torch.from_numpy(L).permute(2, 0, 1)[None].to(config.DEVICE)
    with torch.no_grad():
        ab = model(tensor)[0].cpu().numpy()
    rgb = lab_to_rgb(L, np.transpose(ab, (1, 2, 0)))
    return np.asarray(gray, dtype="uint8"), (rgb * 255).astype("uint8")
