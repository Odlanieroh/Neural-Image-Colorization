import torch
import torch.nn as nn

import config
from data import load_dataloaders
from model import Colorizer


def train(progress_cb=None):
    log = progress_cb or print
    device = config.DEVICE
    log(f"Using device: {device}")
    train_loader, val_loader = load_dataloaders()

    model = Colorizer().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config.LEARNING_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()

    best_val, since_best, best_state = float("inf"), 0, None
    patience = 5

    for epoch in range(config.EPOCHS):
        model.train()
        total, count = 0.0, 0
        for L, ab in train_loader:
            L, ab = L.to(device), ab.to(device)
            opt.zero_grad()
            loss = criterion(model(L), ab)
            loss.backward()
            opt.step()
            total += loss.item() * L.size(0)
            count += L.size(0)
        sched.step()

        model.eval()
        vtotal, vcount = 0.0, 0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(device), ab.to(device)
                vtotal += criterion(model(L), ab).item() * L.size(0)
                vcount += L.size(0)
        val_loss = vtotal / vcount
        log(f"Epoch {epoch + 1}/{config.EPOCHS}  -  loss {total / count:.4f}, val_loss {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                log(f"Early stop: no val improvement for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), config.WEIGHTS_PATH)
    log(f"Saved best model (val_loss {best_val:.4f}) to {config.WEIGHTS_PATH}")
    return model


if __name__ == "__main__":
    train()
