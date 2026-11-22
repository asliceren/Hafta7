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
    return np.clip(img + gurultu, 0.0, 255.0).astype(np.uint8)


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
    img[olasilik < yarim] = 255
    img[(olasilik >= yarim) & (olasilik < gurultu_orani)] = 0
    return img


def dft2d_ve_idft2d_filtrele(
    goruntu: np.ndarray,
    esik_orani: float = 0.02,
) -> np.ndarray:
    img = np.asarray(goruntu, dtype=np.float64)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    M, N = img.shape
    x = np.arange(M, dtype=np.float64)[:, None]
    u = np.arange(M, dtype=np.float64)[None, :]
    W_M = np.exp(-1j * 2.0 * np.pi * x * u / M)

    v = np.arange(N, dtype=np.float64)[:, None]
    y = np.arange(N, dtype=np.float64)[None, :]
    W_N = np.exp(-1j * 2.0 * np.pi * v * y / N)

    F = W_M.T @ img @ W_N.T
    genlik = np.abs(F)
    esik = float(esik_orani) * float(np.max(genlik))

    F_filtreli = np.where(genlik >= esik, F, 0.0 + 0.0j)

    IW_M = np.conj(W_M) / M
    IW_N = np.conj(W_N) / N
    geri = np.real(IW_M @ F_filtreli @ IW_N)
    return np.clip(geri, 0.0, 255.0).astype(np.uint8)


def zncc_hesapla(goruntu_a: np.ndarray, goruntu_b: np.ndarray) -> float:
    a = np.asarray(goruntu_a, dtype=np.float64)
    b = np.asarray(goruntu_b, dtype=np.float64)

    a_ort = a - np.mean(a)
    b_ort = b - np.mean(b)

    payda = np.sqrt(np.sum(a_ort ** 2) * np.sum(b_ort ** 2))
    if payda == 0.0:
        return 0.0

    return float(np.sum(a_ort * b_ort) / payda)


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

geri_gauss = dft2d_ve_idft2d_filtrele(gauss_goruntu, esik_orani=0.02)
geri_sp = dft2d_ve_idft2d_filtrele(sp_goruntu, esik_orani=0.02)

zncc_gauss = zncc_hesapla(orijinal, geri_gauss)
zncc_sp = zncc_hesapla(orijinal, geri_sp)

print(f"Gauss sonrasi IDFT ZNCC:          {zncc_gauss:.4f}")
print(f"Salt & Pepper sonrasi IDFT ZNCC:   {zncc_sp:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(9, 8))

axes[0, 0].imshow(orijinal, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Orijinal Goruntu")
axes[0, 0].axis("off")

axes[0, 1].imshow(gauss_goruntu, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("Gauss Gurultulu Goruntu")
axes[0, 1].axis("off")

axes[1, 0].imshow(geri_gauss, cmap="gray", vmin=0, vmax=255)
axes[1, 0].set_title(f"2D IDFT Sonrasi (ZNCC={zncc_gauss:.4f})")
axes[1, 0].axis("off")

axes[1, 1].imshow(geri_sp, cmap="gray", vmin=0, vmax=255)
axes[1, 1].set_title(f"Salt&Pepper -> 2D IDFT (ZNCC={zncc_sp:.4f})")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("dft2d_ters_donusum_zncc.png", dpi=200, bbox_inches="tight")
plt.show()
