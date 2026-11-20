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
zaman = np.arange(N, dtype=np.float64) / fs
frekanslar = np.arange(N // 2) * (fs / N)

sinyal_1 = 12.0 * np.sin(2.0 * np.pi * 50.0 * zaman)
sinyal_2 = 5.0 * np.sin(2.0 * np.pi * 30.0 * zaman)

gurultulu_1 = gurultu_ekle(sinyal_1, gurultu_orani=0.2, tohum=10)
gurultulu_2 = gurultu_ekle(sinyal_2, gurultu_orani=0.2, tohum=20)

harmonik_1 = dft_harmonik_vektoru(gurultulu_1)
harmonik_2 = dft_harmonik_vektoru(gurultulu_2)

toplam_sinyal = gurultulu_1 + gurultulu_2
harmonik_toplam = dft_harmonik_vektoru(toplam_sinyal)

print("Sinyal 1 (50 Hz, Genlik 12) tepe genligi:", round(float(harmonik_1[50]), 2))
print("Sinyal 2 (30 Hz, Genlik 5) tepe genligi:", round(float(harmonik_2[30]), 2))
print("Toplam sinyal 30 Hz harmonik genligi:", round(float(harmonik_toplam[30]), 2))
print("Toplam sinyal 50 Hz harmonik genligi:", round(float(harmonik_toplam[50]), 2))

fig, axes = plt.subplots(4, 1, figsize=(10, 10))

axes[0].stem(frekanslar, harmonik_1, basefmt=" ")
axes[0].set_title("Gurultulu Sinyal 1 (50 Hz, Genlik=12) - DFT Harmonik Vektoru")
axes[0].set_ylabel("Genlik")
axes[0].set_xlim(0, 100)
axes[0].grid(True, alpha=0.3)

axes[1].stem(frekanslar, harmonik_2, linefmt="C1-", markerfmt="C1o", basefmt=" ")
axes[1].set_title("Gurultulu Sinyal 2 (30 Hz, Genlik=5) - DFT Harmonik Vektoru")
axes[1].set_ylabel("Genlik")
axes[1].set_xlim(0, 100)
axes[1].grid(True, alpha=0.3)

axes[2].plot(zaman, toplam_sinyal, color="purple", linewidth=1.2)
axes[2].set_title("Toplam Gurultulu Sinyal (30 Hz + 50 Hz)")
axes[2].set_xlabel("Zaman (s)")
axes[2].set_ylabel("Genlik")
axes[2].grid(True, alpha=0.3)

axes[3].stem(frekanslar, harmonik_toplam, linefmt="C2-", markerfmt="C2o", basefmt=" ")
axes[3].set_title("Toplam Sinyal - DFT Harmonik Vektoru")
axes[3].set_xlabel("Frekans (Hz)")
axes[3].set_ylabel("Genlik")
axes[3].set_xlim(0, 100)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
