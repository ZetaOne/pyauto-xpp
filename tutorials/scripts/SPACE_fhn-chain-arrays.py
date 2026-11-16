"""
A chain of FitzHugh-Nagumo cells: travelling waves
==================================================

Twenty FitzHugh-Nagumo cells in a line, each coupled to its neighbours by
diffusion of v with strength d, written with XPP's arrays: %[1..20] ... %
repeats the lines with [j] = 1, 2, ..., 20, and v0 = v1, v21 = v20 make the
ends no-flux.  Cell 1 gets an input i1 and fires repeatedly; each spike
travels down the chain as a wave.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. draws the space-time picture of the waves
  3. compares the firing period at cell 1 and at cell 20 (every spike that
     starts at cell 1 reaches the far end)
  4. measures the wave speed (cells per unit time) against the coupling d and
     fits a power law: for a wide front the chain behaves like a cable,
     v_t = d v_xx + ..., whose wave speed grows like sqrt(d)
  5. checks the results

Run:   python SPACE_fhn-chain-arrays.py
Figures go to figures/SPACE_fhn-chain-arrays/.

Model from the examples of XPP Studio (Peter Ashwin and Bard Ermentrout).
Every number is computed by XPPAUT 8.1 and its AUTO through pyxpp.
"""
# %% setup
from pathlib import Path

import matplotlib
# figures are saved to figures/ and shown at the end (plt.show());
# run_all.py runs without windows (MPLBACKEND=Agg)
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
from pyxpp.builder import ModelBuilder

try:
    HERE = Path(__file__).resolve().parent
    NAME = Path(__file__).stem
except NameError:            # run as a notebook in Jupyter: the kernel starts in this folder
    HERE = Path.cwd()
    NAME = "SPACE_fhn-chain-arrays"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)
N = 20


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


def crossings(tr, col, level=0.0):
    """Times where column col rises through level, interpolated between output steps."""
    x, t = tr[col], tr.t
    k = np.flatnonzero((x[:-1] < level) & (x[1:] >= level))
    return t[k] + (level - x[k]) / (x[k + 1] - x[k]) * (t[k + 1] - t[k])


# %% 1. the model
mb = (ModelBuilder("fhn_chain")
      .comment("A chain of 20 diffusively coupled FitzHugh-Nagumo cells")
      .begin_block("1..20")
      .ode("v[j]", "v[j]-v[j]^3/3-w[j]+i1*heav(1.5-[j])+d*(v[j+1]-2*v[j]+v[j-1])")
      .end_block()
      .ode("w[1..20]", "eps*(v[j]+a-b*w[j])")
      .fixed("v0", "v1")                    # no-flux ends
      .fixed("v21", "v20")
      .parameter(i1=1, d=1, eps=0.08, a=0.7, b=0.8)
      .initial({"v[1..20]": -1.2})
      .initial({"w[1..20]": -0.62})
      .option(total=200, dt=0.05))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
cells = [f"v{j}" for j in range(1, N + 1)]          # the columns v1 ... v20

# %% 2. space-time picture at d = 1
tr = m.run(keep=False, total=200)
V = np.array([tr[c] for c in cells])                # cells x time
fig, ax = plt.subplots(figsize=(7.5, 3.8))
im = ax.imshow(V, aspect="auto", origin="lower", cmap="magma", interpolation="nearest",
               extent=(tr.t[0], tr.t[-1], 0.5, N + 0.5))
fig.colorbar(im, ax=ax, label="v")
ax.set(xlabel="t", ylabel="cell", yticks=[1, 5, 10, 15, 20], title="Waves along the chain (d = 1, input to cell 1)")
fig.tight_layout()
fig.savefig(FIG / "space_time.png", dpi=130)

# %% 3. firing period at the two ends
tr = m.run(keep=False, total=800)
sp1, sp20 = crossings(tr, "v1"), crossings(tr, "v20")
P1, P20 = np.diff(sp1)[-5:].mean(), np.diff(sp20)[-5:].mean()
transit = sp20[0] - sp1[0]
n1 = np.sum(sp1 < tr.t[-1] - transit)                # spikes at cell 1 that had time to arrive
print(f"\nperiod at cell 1 {P1:.4f}, at cell 20 {P20:.4f}; transit time {transit:.2f};"
      f" spikes {n1} at cell 1 (with time to arrive), {len(sp20)} at cell 20")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(tr.t, tr["v1"], lw=1, label="cell 1")
ax.plot(tr.t, tr["v20"], lw=1, label="cell 20")
ax.set(xlim=(0, 300), xlabel="t", ylabel="v", title=f"Same period at both ends ({P1:.2f})")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "ends.png", dpi=130)

# %% 4. wave speed against the coupling
# start a single wave by kicking cells 1-3 (no input), then time it across cells 5..16,
# away from the kick and from the far end
D = np.array([0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8])
kick = {f"v{j}": 2.0 for j in (1, 2, 3)}
speed, worst = [], []
for d in D:
    tr = m.run(keep=False, d=d, i1=0, total=150, **kick)
    js = np.arange(5, 17)
    arrive = np.array([crossings(tr, f"v{j}")[0] for j in js])
    slope, icpt = np.polyfit(js, arrive, 1)                  # time per cell
    speed.append(1 / slope)
    worst.append(np.abs(arrive - (slope * js + icpt)).max())
speed, worst = np.array(speed), np.array(worst)
fit = D >= 2                                     # wide fronts: several cells across
p, logA = np.polyfit(np.log(D[fit]), np.log(speed[fit]), 1)
print("\n   d    speed   speed/sqrt(d)")
for d, c in zip(D, speed):
    print(f"  {d:4.1f}  {c:.4f}  {c / np.sqrt(d):.4f}")
print(f"power law over d >= 2: speed = {np.exp(logA):.4f} d^{p:.4f}")

fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.loglog(D[fit], speed[fit], "o", color="tab:blue", label="measured (used in the fit)")
ax.loglog(D[~fit], speed[~fit], "o", mfc="none", color="tab:blue", label="measured")
dd = np.geomspace(0.15, 10, 50)
ax.loglog(dd, np.exp(logA) * dd**p, "-", color="tab:red", lw=1, label=f"fit (d >= 2): exponent {p:.3f}")
ax.loglog(dd, speed[D == 4][0] * np.sqrt(dd / 4), "k--", lw=1, label="sqrt(d) through d = 4")
ax.set(xlabel="coupling d", ylabel="wave speed (cells per unit time)", title="Wave speed against coupling")
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([0.3, 0.5, 1, 2])
ax.set_xticks([0.2, 0.5, 1, 2, 5, 10])
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "speed_vs_d.png", dpi=130)

# %% 5. checks
print("\nchecks:")
# interpolated crossing times are good to well under 0.01 with dt = 0.05
check("period at cell 20 = period at cell 1", P20, P1, 0.01)
check("every spike reaches cell 20 (spike counts)", len(sp20), n1, 0)
# a wave at constant speed arrives on a straight line in (cell, t), to within an output step
check("constant speed: largest miss of a straight line in t (d = 1)", worst[D == 1][0], 0, 0.05)
# cable limit: speed = c* sqrt(d).  On the lattice, c/sqrt(d) still creeps up towards c*
# with d (the front spans only a few cells); over d = 2..8 the printed table shows a drift of
# about 2%, which raises the fitted exponent by about 0.02: allow 0.03
check("power-law exponent of speed against d", p, 0.5, 0.03)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
