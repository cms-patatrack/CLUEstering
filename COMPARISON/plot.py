import pandas as pd
import matplotlib.pyplot as plt

sp = pd.read_csv("gaussian_2048_new.csv")
bf = pd.read_csv("gaussian_2048_old.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, df, title in zip(axes, [sp, bf], ["New: spectral", "Old: brute force"]):
    sc = ax.scatter(df.x, df.y, c=df.rho, cmap="viridis", s=8)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax)
plt.tight_layout()
plt.savefig("density_comparison_gaussian_2048.png", dpi=150)