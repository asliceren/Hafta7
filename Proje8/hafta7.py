import cv2
import matplotlib.pyplot as plt
import numpy as np


def gauss_gurultu_ekle(
    goruntu: np.ndarray,
    gurultu_kuvveti: float = 25.0,
    tohum: int | None = 42,
) -> np.ndarray:
    img = np.asarray(goruntu, dtype=np.float64)
    rng = np.random.default_rng(tohum)
    gurultu = rng.normal(loc=0.0, scale=float(gurultu_kuvveti), size=img.shape)
    sonuc = np.clip(img + gurultu, 0.0, 255.0)
    return sonuc.astype(np.uint8)


def salt_and_pepper_gurultu_ekle(
    goruntu: np.ndarray,
    gurultu_orani: float = 0.05,
    tohum: int | None = 42,
) -> np.ndarray:
    if not (0.0 <= gurultu_orani <= 1.0):
        raise ValueError("Gurultu orani [0, 1] araliginda olmalidir.")

    img = np.asarray(goruntu, dtype=np.uint8).copy()
    rng = np.random.default_rng(tohum)
    olasilik = rng.random(size=img.shape[:2])

    yarim = gurultu_orani / 2.0
    tuz_maske = olasilik < yarim
    biber_maske = (olasilik >= yarim) & (olasilik < gurultu_orani)

    img[tuz_maske] = 255
    img[biber_maske] = 0
    return img


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
orijinal = (
    127.5
    + 60.0 * np.sin(2.0 * np.pi * 6.0 * xx / M)
    + 40.0 * np.cos(2.0 * np.pi * 12.0 * yy / N)
)
orijinal = np.clip(orijinal, 0, 255).astype(np.uint8)

gauss_goruntu = gauss_gurultu_ekle(orijinal, gurultu_kuvveti=25.0, tohum=42)
sp_goruntu = salt_and_pepper_gurultu_ekle(orijinal, gurultu_orani=0.05, tohum=42)

harm_orijinal = dft2d_harmonik_matrisi(orijinal, merkeze_kaydir=True)
harm_gauss = dft2d_harmonik_matrisi(gauss_goruntu, merkeze_kaydir=True)
harm_sp = dft2d_harmonik_matrisi(sp_goruntu, merkeze_kaydir=True)

log_orijinal = np.log1p(harm_orijinal)
log_gauss = np.log1p(harm_gauss)
log_sp = np.log1p(harm_sp)

fark_gauss = np.mean(np.abs(harm_gauss - harm_orijinal))
fark_sp = np.mean(np.abs(harm_sp - harm_orijinal))

print(f"Gauss harmonik ortalama mutlak fark:          {fark_gauss:.2f}")
print(f"Salt & Pepper harmonik ortalama mutlak fark:   {fark_sp:.2f}")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))

axes[0, 0].imshow(orijinal, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Orijinal Goruntu")
axes[0, 0].axis("off")

axes[0, 1].imshow(gauss_goruntu, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("Gauss Gurultulu (sigma=25)")
axes[0, 1].axis("off")

axes[0, 2].imshow(sp_goruntu, cmap="gray", vmin=0, vmax=255)
axes[0, 2].set_title("Salt & Pepper Gurultulu (oran=0.05)")
axes[0, 2].axis("off")

axes[1, 0].imshow(log_orijinal, cmap="inferno")
axes[1, 0].set_title("Orijinal - 2D DFT Harmonik Matris")
axes[1, 0].axis("off")

axes[1, 1].imshow(log_gauss, cmap="inferno")
axes[1, 1].set_title("Gauss - 2D DFT Harmonik Matris")
axes[1, 1].axis("off")

axes[1, 2].imshow(log_sp, cmap="inferno")
axes[1, 2].set_title("Salt & Pepper - 2D DFT Harmonik Matris")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("dft2d_gurultu_kiyas.png", dpi=200, bbox_inches="tight")
plt.show()
