import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data import load_dataloaders, lab_to_rgb
from inference import load_model


def evaluate(model=None, report_path="visual_report.png"):
    _, val_loader = load_dataloaders()
    if model is None:
        model = load_model()
    model.eval()

    psnrs, samples = [], []
    with torch.no_grad():
        for L, ab in val_loader:
            pred = model(L.to(config.DEVICE)).cpu().numpy()
            Lnp, abnp = L.numpy(), ab.numpy()
            for i in range(L.size(0)):
                Li = np.transpose(Lnp[i], (1, 2, 0))
                gt = lab_to_rgb(Li, np.transpose(abnp[i], (1, 2, 0)))
                pr = lab_to_rgb(Li, np.transpose(pred[i], (1, 2, 0)))
                # RGB values are in [0, 1], making 1.0 the peak value in the PSNR formula.
                mse = float(np.mean((gt - pr) ** 2))
                psnrs.append(10 * np.log10(1.0 / max(mse, 1e-10)))
                if len(samples) < 10:
                    samples.append((Li[:, :, 0], gt, pr))

    mean_psnr = float(np.mean(psnrs))

    fig, axs = plt.subplots(3, 10, figsize=(22, 7))
    for i, (gray, gt, pr) in enumerate(samples):
        axs[0, i].imshow(gray, cmap="gray")
        axs[1, i].imshow(gt)
        axs[2, i].imshow(pr)
        for r in range(3):
            axs[r, i].axis("off")
    for r, lbl in enumerate(["Input", "Ground truth", "Prediction"]):
        axs[r, 0].set_ylabel(lbl, size=12)
    plt.tight_layout()
    plt.savefig(report_path, dpi=150, bbox_inches="tight")
    pdf_path = report_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    metrics = {
        "mean_psnr_db": round(mean_psnr, 4),
        "validation_images": len(psnrs),
        "device": str(config.DEVICE),
        "report_png": report_path,
        "report_pdf": pdf_path,
    }
    with open("evaluation_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    return mean_psnr, pdf_path


if __name__ == "__main__":
    score, pdf = evaluate()
    print(f"Mean PSNR: {score:.4f} dB")
    print(f"PDF report: {pdf}")
    print("Metrics: evaluation_metrics.json")
