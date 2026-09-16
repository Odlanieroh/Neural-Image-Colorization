import torch

IMG_SIZE       = 128
BATCH_SIZE     = 32
EPOCHS         = 20
LEARNING_RATE  = 1e-4
FREEZE_ENCODER = False
WEIGHTS_PATH   = "colorizer.pth"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
