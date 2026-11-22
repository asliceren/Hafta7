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
    gurultu = rng.normal(loc=0.0, scale=gurultu_orani * genlik, size=vektor.shape)
    return vektor + gurultu


def dft_ve_idft_matrisleri(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Vektor boyutu 2'nin kuvveti olmalidir, gelen boyut: {n}")

    k = np.arange(n, dtype=np.float64)[:, None]
    m = np.arange(n, dtype=np.float64)[None, :]
    dft_matrisi = np.exp(-1j * 2.0 * np.pi * k * m / n)
    idft_matrisi = np.exp(1j * 2.0 * np.pi * k * m / n) / n
    return dft_matrisi, idft_matrisi


def frekans_filtresi_uygula(
    vektor: np.ndarray,
    esik_degeri: float,
    normalize_genlik_kullan: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(vektor, dtype=np.float64)
    n = x.shape[0]
    dft_matrisi, idft_matrisi = dft_ve_idft_matrisleri(n)

    X = dft_matrisi @ x

    if normalize_genlik_kullan:
        harmonik_genlik = np.abs(X) / n
    else:
        harmonik_genlik = np.abs(X)

    X_filtreli = np.where(harmonik_genlik >= esik_degeri, X, 0.0 + 0.0j)
    yeniden_uretilen = np.real(idft_matrisi @ X_filtreli)

    harmonik_cikti = np.abs(X_filtreli[: n // 2]) / n
    if n // 2 > 1:
        harmonik_cikti[1:] *= 2.0

    return yeniden_uretilen, harmonik_cikti, X_filtreli


N = 256
fs = 256.0
zaman = np.arange(N, dtype=np.float64) / fs
frekanslar = np.arange(N // 2) * (fs / N)

temiz_1 = 12.0 * np.sin(2.0 * np.pi * 50.0 * zaman)
temiz_2 = 5.0 * np.sin(2.0 * np.pi * 30.0 * zaman)
temiz_toplam = temiz_1 + temiz_2

gurultulu_1 = gurultu_ekle(temiz_1, gurultu_orani=0.2, tohum=10)
gurultulu_2 = gurultu_ekle(temiz_2, gurultu_orani=0.2, tohum=20)
gurultulu_toplam = gurultulu_1 + gurultulu_2

esik = 1.5

geri_1, harm_1, _ = frekans_filtresi_uygula(gurultulu_1, esik_degeri=esik)
geri_2, harm_2, _ = frekans_filtresi_uygula(gurultulu_2, esik_degeri=esik)
geri_toplam, harm_toplam, _ = frekans_filtresi_uygula(gurultulu_toplam, esik_degeri=esik)

hata_1 = np.sqrt(np.mean((temiz_1 - geri_1) ** 2))
hata_2 = np.sqrt(np.mean((temiz_2 - geri_2) ** 2))
hata_toplam = np.sqrt(np.mean((temiz_toplam - geri_toplam) ** 2))

print(f"Sinyal 1 (50 Hz, A=12) RMSE: {hata_1:.4f}")
print(f"Sinyal 2 (30 Hz, A=5)  RMSE: {hata_2:.4f}")
print(f"Toplam Sinyal RMSE:          {hata_toplam:.4f}")

fig, axes = plt.subplots(3, 2, figsize=(12, 9))

axes[0, 0].plot(zaman[:64], gurultulu_1[:64], color="gray", alpha=0.6, label="Gurultulu")
axes[0, 0].plot(zaman[:64], geri_1[:64], color="blue", linewidth=1.8, label="Filtreli (50 Hz)")
axes[0, 0].plot(zaman[:64], temiz_1[:64], "--", color="green", alpha=0.8, label="Temiz")
axes[0, 0].set_title("Sinyal 1 (50 Hz, A=12) - Yeniden Uretim")
axes[0, 0].legend(loc="upper right", fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].stem(frekanslar, harm_1, basefmt=" ")
axes[0, 1].axhline(esik, color="red", linestyle="--", linewidth=1, label=f"Esik={esik}")
axes[0, 1].set_title("Sinyal 1 - Filtrelenmis Harmonikler")
axes[0, 1].set_xlim(0, 100)
axes[0, 1].legend(loc="upper right", fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(zaman[:64], gurultulu_2[:64], color="gray", alpha=0.6, label="Gurultulu")
axes[1, 0].plot(zaman[:64], geri_2[:64], color="orange", linewidth=1.8, label="Filtreli (30 Hz)")
axes[1, 0].plot(zaman[:64], temiz_2[:64], "--", color="green", alpha=0.8, label="Temiz")
axes[1, 0].set_title("Sinyal 2 (30 Hz, A=5) - Yeniden Uretim")
axes[1, 0].legend(loc="upper right", fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].stem(frekanslar, harm_2, linefmt="C1-", markerfmt="C1o", basefmt=" ")
axes[1, 1].axhline(esik, color="red", linestyle="--", linewidth=1, label=f"Esik={esik}")
axes[1, 1].set_title("Sinyal 2 - Filtrelenmis Harmonikler")
axes[1, 1].set_xlim(0, 100)
axes[1, 1].legend(loc="upper right", fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

axes[2, 0].plot(zaman[:64], gurultulu_toplam[:64], color="gray", alpha=0.6, label="Toplam Gurultulu")
axes[2, 0].plot(zaman[:64], geri_toplam[:64], color="purple", linewidth=1.8, label="Toplam Filtreli")
axes[2, 0].plot(zaman[:64], temiz_toplam[:64], "--", color="green", alpha=0.8, label="Toplam Temiz")
axes[2, 0].set_xlabel("Zaman (s)")
axes[2, 0].set_title("Birlesik Sinyal (30 Hz + 50 Hz) - Yeniden Uretim")
axes[2, 0].legend(loc="upper right", fontsize=8)
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].stem(frekanslar, harm_toplam, linefmt="C2-", markerfmt="C2o", basefmt=" ")
axes[2, 1].axhline(esik, color="red", linestyle="--", linewidth=1, label=f"Esik={esik}")
axes[2, 1].set_xlabel("Frekans (Hz)")
axes[2, 1].set_title("Birlesik Sinyal - Filtrelenmis Harmonikler")
axes[2, 1].set_xlim(0, 100)
axes[2, 1].legend(loc="upper right", fontsize=8)
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("frekans_filtresi.png", dpi=200, bbox_inches="tight")
plt.show()
