import argparse
import pandas as pd
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--spectral", required=True)
ap.add_argument("-b", "--bruteforce", required=True)
ap.add_argument("-d", "--dotsize", type=int, default=8)
ap.add_argument("-o", "--out", required=True)
args = ap.parse_args()

sp, bf = pd.read_csv(args.spectral), pd.read_csv(args.bruteforce)
vmax = max(sp.rho.max(), bf.rho.max())

fig = plt.figure(figsize=(10, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[1, :], sharex=ax1, sharey=ax1)

sc1 = ax1.scatter(sp.x, sp.y, c=sp.rho, cmap="viridis", s=args.dotsize, vmin=0.0, vmax=vmax)
sc2 = ax2.scatter(bf.x, bf.y, c=bf.rho, cmap="viridis", s=args.dotsize, vmin=0.0, vmax=vmax)
sc3 = ax3.scatter(bf.x, bf.y, c=sp.rho - bf.rho, cmap="inferno", s=args.dotsize)

titles = ["New: spectral", "Old: brute force", "spectral - brute force"]
for ax, title in zip([ax1, ax2, ax3], titles):
    ax.set_aspect('equal', 'box')
    ax.set_title(title)

fig.colorbar(sc1, ax=[ax1, ax2], label='rho')
fig.colorbar(sc3, ax=ax3, label='Delta rho')

fig.savefig(args.out)