import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_summed_png(h5_path: str, out_png: str | None = None, dataset: str = "/images/summed"):
    h5_path = str(h5_path)
    with h5py.File(h5_path, "r") as f:
        if dataset not in f:
            raise KeyError(f"{dataset} not found in {h5_path}")
        img = np.array(f[dataset], dtype=np.float32)

    if out_png is None:
        out_png = str(Path(h5_path).with_suffix(".png"))

    plt.figure()
    plt.imshow(img, origin="lower")
    plt.colorbar()
    plt.title(Path(h5_path).name + " : " + dataset)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png
