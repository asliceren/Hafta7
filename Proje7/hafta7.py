import cv2
import matplotlib.pyplot as plt
import numpy as np


def dft2d_harmonik_matrisi(goruntu: np.ndarray, merkeze_kaydir: bool = True) -> np.ndarray:
    img = np.asarray(goruntu, dtype=np.float64)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    if img.ndim != 2:
        raise ValueError("Girdi 2 boyutlu gri tonlamali bir goruntu olmalidir.")

    M, N = img.shape
    x = np.arange(M, dtype=np.float64)[:, None]
    u = np.arange(M, dtype=np.float64)[None, :]
    W_M = np.exp(-1j * 2.0 * np.pi * x * u / M)

    v = np.arange(N, dtype=np.float64)[:, None]
    y = np.arange(N, dtype=np.float64)[None, :]
    W_N = np.exp(-1j * 2.0 * np.pi * v * y / N)

    F = W_M.T @ img @ W_N.T
    harmonik = np.abs(F)

    if merkeze_kaydir:
        harmonik = np.roll(harmonik, shift=(M // 2, N // 2), axis=(0, 1))

    return harmonik


M, N = 128, 128
yy, xx = np.meshgrid(np.arange(N), np.arange(M))
test_goruntu = (
    127.5
    + 60.0 * np.sin(2.0 * np.pi * 6.0 * xx / M)
    + 40.0 * np.cos(2.0 * np.pi * 12.0 * yy / N)
)
test_goruntu = np.clip(test_goruntu, 0, 255).astype(np.uint8)

harmonik_matris = dft2d_harmonik_matrisi(test_goruntu, merkeze_kaydir=True)
log_spektrum = np.log1p(harmonik_matris)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

axes[0].imshow(test_goruntu, cmap="gray")
axes[0].set_title("Girdi Goruntusu")
axes[0].axis("off")

axes[1].imshow(log_spektrum, cmap="inferno")
axes[1].set_title("2D DFT Harmonik Matrisi (Log Spektrum)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("dft2d_harmonik.png", dpi=200, bbox_inches="tight")
plt.show()
