import matplotlib.pyplot as plt
import numpy as np


def doldur_sinus_vektoru(
    vektor: np.ndarray,
    genlik: float,
    faz_radyan: float = 0.0,
    frekans: float | None = None,
    periyot: float | None = None,
    ornekleme_hizi: float = 256.0,
) -> np.ndarray:
    """
    2'nin kuvveti uzunlugundaki 'vektor' dizisini sinus degerleriyle doldurur.
    frekans veya periyot parametresinden en az biri verilmelidir.
    """
    n = vektor.shape[0]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Vektor boyutu 2'nin kuvveti olmalidir, gelen boyut: {n}")

    if frekans is None and periyot is None:
        raise ValueError("Frekans veya periyot degerlerinden en az biri verilmelidir.")

    if frekans is None:
        if periyot <= 0:
            raise ValueError("Periyot pozitif olmalidir.")
        frekans = 1.0 / periyot

    zaman = np.arange(n, dtype=np.float64) / float(ornekleme_hizi)
    vektor[:] = genlik * np.sin(2.0 * np.pi * frekans * zaman + faz_radyan)
    return vektor


# Vektor boyutu 2'nin kuvveti (orn: 2^8 = 256)
N = 256
sinus_vektoru = np.zeros(N, dtype=np.float64)

# Fonksiyonu frekans ile doldurma (ornegin: genlik=2.5, frekans=4 Hz, faz=pi/4)
doldur_sinus_vektoru(
    vektor=sinus_vektoru,
    genlik=2.5,
    faz_radyan=np.pi / 4.0,
    frekans=4.0,
    ornekleme_hizi=256.0,
)

# Gorsellestirme: Vektor indeksleri ve zamana bagli sinuzoidal dalga
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)

# 1. Vektor elemanlari (0 .. N-1)
axes[0].plot(np.arange(N), sinus_vektoru, color="royalblue", linewidth=1.5)
axes[0].set_title("Doldurulan Sinus Vektoru (Boyut = 2^8 = 256)")
axes[0].set_xlabel("Vektor Indeksi (n)")
axes[0].set_ylabel("Genlik")
axes[0].grid(True, alpha=0.3)

# 2. Zaman ekseninde sinuzoidal dalga
ornekleme_hizi = 256.0
t = np.arange(N) / ornekleme_hizi
axes[1].plot(t, sinus_vektoru, color="crimson", linewidth=1.5, label="A=2.5, f=4 Hz, phi=pi/4")
axes[1].set_title("Sinuzoidal Dalga Gorunumu")
axes[1].set_xlabel("Zaman (s)")
axes[1].set_ylabel("Genlik")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
