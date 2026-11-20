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


N = 256
zaman = np.arange(N, dtype=np.float64) / 256.0
temiz_vektor = 2.5 * np.sin(2.0 * np.pi * 4.0 * zaman + np.pi / 4.0)

gurultu_orani = 0.2
gurultulu_vektor = gurultu_ekle(temiz_vektor, gurultu_orani=gurultu_orani)

plt.figure(figsize=(9, 4.5))
plt.plot(zaman, temiz_vektor, label="Temiz Sinyal", linewidth=1.5)
plt.plot(zaman, gurultulu_vektor, label=f"Gurultulu Sinyal (oran={gurultu_orani})", alpha=0.85)
plt.title("Vektore Rastgele Gurultu Ekleme")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
