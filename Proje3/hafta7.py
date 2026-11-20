import matplotlib.pyplot as plt
import numpy as np


def gurultu_ekle(
    vektor: np.ndarray,
    gurultu_orani: float,
    tohum: int | None = 42,
) -> np.ndarray:
    if not (0.0 <= gurultu_orani <= 1.0):
        raise ValueError("Gurultu orani [0, 1] araliginda olmalidir.")

    rng = np.random.default_rng(tohum)
    genlik = np.max(np.abs(vektor))
    gurultu_sapmasi = gurultu_orani * genlik
    gurultu = rng.normal(loc=0.0, scale=gurultu_sapmasi, size=vektor.shape)
    return vektor + gurultu


def gauss_filtre_1d(sinyal: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    yaricap = int(np.ceil(3.0 * sigma))
    x = np.arange(-yaricap, yaricap + 1, dtype=np.float64)
    cekirdek = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    cekirdek /= np.sum(cekirdek)

    dolgulu = np.pad(sinyal, pad_width=yaricap, mode="edge")
    return np.convolve(dolgulu, cekirdek, mode="valid")


def medyan_filtre_1d(sinyal: np.ndarray, pencere: int = 5) -> np.ndarray:
    if pencere % 2 == 0 or pencere < 1:
        raise ValueError("Pencere boyutu 1'den buyuk tek sayi olmalidir.")

    yaricap = pencere // 2
    dolgulu = np.pad(sinyal, pad_width=yaricap, mode="edge")
    cikti = np.empty_like(sinyal, dtype=np.float64)

    for i in range(len(sinyal)):
        cikti[i] = np.median(dolgulu[i : i + pencere])

    return cikti


N = 256
zaman = np.arange(N, dtype=np.float64) / 256.0
temiz_vektor = 2.5 * np.sin(2.0 * np.pi * 4.0 * zaman + np.pi / 4.0)

gurultu_orani = 0.2
gurultulu_vektor = gurultu_ekle(temiz_vektor, gurultu_orani=gurultu_orani)

gauss_temiz = gauss_filtre_1d(gurultulu_vektor, sigma=2.0)
medyan_temiz = medyan_filtre_1d(gurultulu_vektor, pencere=5)

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

axes[0].plot(zaman, gurultulu_vektor, label="Gurultulu Sinyal", alpha=0.6, color="gray")
axes[0].plot(zaman, gauss_temiz, label="Gauss Filtre (sigma=2.0)", linewidth=2, color="blue")
axes[0].plot(zaman, temiz_vektor, "--", label="Orijinal Temiz Sinyal", alpha=0.8, color="green")
axes[0].set_title("Gauss Filtresi Sonucu")
axes[0].set_ylabel("Genlik")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

axes[1].plot(zaman, gurultulu_vektor, label="Gurultulu Sinyal", alpha=0.6, color="gray")
axes[1].plot(zaman, medyan_temiz, label="Medyan Filtre (pencere=5)", linewidth=2, color="red")
axes[1].plot(zaman, temiz_vektor, "--", label="Orijinal Temiz Sinyal", alpha=0.8, color="green")
axes[1].set_title("Medyan Filtresi Sonucu")
axes[1].set_xlabel("Zaman (s)")
axes[1].set_ylabel("Genlik")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()