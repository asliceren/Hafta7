import matplotlib.pyplot as plt
import numpy as np


def dft_harmonik_vektoru(vektor: np.ndarray) -> np.ndarray:
    x = np.asarray(vektor, dtype=np.float64)
    n = x.shape[0]

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Vektor boyutu 2'nin kuvveti olmalidir, gelen boyut: {n}")

    k = np.arange(n, dtype=np.float64)[:, None]
    m = np.arange(n, dtype=np.float64)[None, :]
    dft_matrisi = np.exp(-1j * 2.0 * np.pi * k * m / n)

    X = dft_matrisi @ x
    yarim = n // 2
    harmonik = np.abs(X[:yarim]) / n
    if yarim > 1:
        harmonik[1:] *= 2.0
    return harmonik


def gurultu_ekle(
    vektor: np.ndarray,
    gurultu_orani: float,
    tohum: int | None = 42,
) -> np.ndarray:
    if not (0.0 <= gurultu_orani <= 1.0):
        raise ValueError("Gurultu orani [0, 1] araliginda olmalidir.")

    rng = np.random.default_rng(tohum)
    genlik = np.max(np.abs(vektor))
    gurultu = rng.normal(loc=0.0, scale=gurultu_orani * genlik, size=vektor.shape)
    return vektor + gurultu


N = 256
fs = 256.0
f_sinyal = 4.0
zaman = np.arange(N, dtype=np.float64) / fs

temiz_vektor = 2.5 * np.sin(2.0 * np.pi * f_sinyal * zaman + np.pi / 4.0)
gurultulu_vektor = gurultu_ekle(temiz_vektor, gurultu_orani=0.2)

harmonik_temiz = dft_harmonik_vektoru(temiz_vektor)
harmonik_gurultulu = dft_harmonik_vektoru(gurultulu_vektor)

frekanslar = np.arange(N // 2) * (fs / N)

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

axes[0].stem(frekanslar, harmonik_temiz, basefmt=" ")
axes[0].set_title("Gurultusuz Sinyal - DFT Harmonik Vektoru")
axes[0].set_ylabel("Genlik")
axes[0].set_xlim(0, 64)
axes[0].grid(True, alpha=0.3)

axes[1].stem(frekanslar, harmonik_gurultulu, linefmt="C1-", markerfmt="C1o", basefmt=" ")
axes[1].set_title("Gurultulu Sinyal (oran=0.2) - DFT Harmonik Vektoru")
axes[1].set_xlabel("Frekans (Hz)")
axes[1].set_ylabel("Genlik")
axes[1].set_xlim(0, 64)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
