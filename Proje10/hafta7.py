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


def dft2d_matrisleri(M: int, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(M, dtype=np.float64)[:, None]
    u = np.arange(M, dtype=np.float64)[None, :]
    W_M = np.exp(-1j * 2.0 * np.pi * x * u / M)

    v = np.arange(N, dtype=np.float64)[:, None]
    y = np.arange(N, dtype=np.float64)[None, :]
    W_N = np.exp(-1j * 2.0 * np.pi * v * y / N)

    IW_M = np.conj(W_M) / M
    IW_N = np.conj(W_N) / N
    return W_M, W_N, IW_M, IW_N


def merkez_frekans_uzakligi(M: int, N: int) -> np.ndarray:
    u_ind = np.arange(M, dtype=np.float64) - (M // 2)
    v_ind = np.arange(N, dtype=np.float64) - (N // 2)
    vv, uu = np.meshgrid(v_ind, u_ind)
    return np.sqrt(uu ** 2 + vv ** 2)


def filtre_maskesi_olustur(
    M: int,
    N: int,
    filtre_tipi: str,
    d_dusuk: float = 12.0,
    d_yuksek: float = 35.0,
) -> np.ndarray:
    uzaklik = merkez_frekans_uzakligi(M, N)
    tip = filtre_tipi.lower().strip()

    if tip == "low-pass":
        maske = (uzaklik <= d_dusuk).astype(np.float64)
    elif tip == "high-pass":
        maske = (uzaklik > d_dusuk).astype(np.float64)
    elif tip == "band-pass":
        maske = ((uzaklik >= d_dusuk) & (uzaklik <= d_yuksek)).astype(np.float64)
    elif tip == "band-reject":
        maske = ~((uzaklik >= d_dusuk) & (uzaklik <= d_yuksek))
        maske = maske.astype(np.float64)
    else:
        raise ValueError("Gecersiz filtre tipi: low-pass, high-pass, band-pass veya band-reject secin.")

    return np.roll(maske, shift=(-M // 2, -N // 2), axis=(0, 1))


def frekans_filtresi_2d(
    goruntu: np.ndarray,
    filtre_tipi: str,
    d_dusuk: float = 12.0,
    d_yuksek: float = 35.0,
) -> np.ndarray:
    img = np.asarray(goruntu, dtype=np.float64)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    M, N = img.shape
    W_M, W_N, IW_M, IW_N = dft2d_matrisleri(M, N)

    F = W_M.T @ img @ W_N.T
    maske = filtre_maskesi_olustur(M, N, filtre_tipi, d_dusuk=d_dusuk, d_yuksek=d_yuksek)
    F_filtreli = F * maske

    geri = np.real(IW_M @ F_filtreli @ IW_N)
    return np.clip(geri, 0.0, 255.0).astype(np.uint8)


M, N = 128, 128
yy, xx = np.meshgrid(np.arange(N), np.arange(M))
orijinal = (
    127.5
    + 60.0 * np.sin(2.0 * np.pi * 6.0 * xx / M)
    + 40.0 * np.cos(2.0 * np.pi * 12.0 * yy / N)
)
orijinal[32:96, 32:96] = np.clip(orijinal[32:96, 32:96] + 55.0, 0, 255)
orijinal = np.clip(orijinal, 0, 255).astype(np.uint8)

gauss_img = gauss_gurultu_ekle(orijinal, gurultu_kuvveti=25.0, tohum=42)
sp_img = salt_and_pepper_gurultu_ekle(orijinal, gurultu_orani=0.05, tohum=42)

filtreler = ["low-pass", "high-pass", "band-pass", "band-reject"]

fig, axes = plt.subplots(3, 5, figsize=(16, 9.5))

satir_girdileri = [
    ("Orijinal", orijinal),
    ("Gauss Gurultu", gauss_img),
    ("Salt & Pepper", sp_img),
]

for r_idx, (etiket, girdi_img) in enumerate(satir_girdileri):
    axes[r_idx, 0].imshow(girdi_img, cmap="gray", vmin=0, vmax=255)
    axes[r_idx, 0].set_title(f"{etiket} (Ortalama: {np.mean(girdi_img):.1f})")
    axes[r_idx, 0].axis("off")

    for c_idx, f_tip in enumerate(filtreler, start=1):
        cikti_img = frekans_filtresi_2d(girdi_img, f_tip, d_dusuk=12.0, d_yuksek=35.0)
        axes[r_idx, c_idx].imshow(cikti_img, cmap="gray", vmin=0, vmax=255)
        axes[r_idx, c_idx].set_title(f"{f_tip} (Ort: {np.mean(cikti_img):.1f})")
        axes[r_idx, c_idx].axis("off")

plt.tight_layout()
plt.savefig("frekans_filtreleri_2d.png", dpi=200, bbox_inches="tight")
plt.show()
