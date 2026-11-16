"""
FitzHugh-Nagumo: an excitable membrane
======================================

A two-variable caricature of a neuron: a fast voltage-like variable v with a
cubic nullcline and a slow recovery variable w.  For small input i there is one
stable rest state, and a spike only after a big enough kick; as i grows the
rest state loses stability in a Hopf bifurcation and the cell fires repeatedly.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. draws the phase plane: XPP's direction field, nullclines, the equilibrium
     with its eigenvalues, and trajectories
  3. shows excitability: the peak voltage against the size of a kick
  4. follows the equilibrium in i with AUTO, finds the two Hopf points and
     their first Lyapunov coefficients, and continues the periodic orbits
  5. checks the results against values known for this model

Run:   python 2D-FLOWS_fitzhugh-nagumo.py
Figures go to figures/2D-FLOWS_fitzhugh-nagumo/.

Model from the examples of XPP Studio (Peter Ashwin and Bard Ermentrout).
Every number is computed by XPPAUT 8.1 and its AUTO through pyxpp.
"""
# %% setup
from pathlib import Path

import matplotlib
# figures are saved to figures/ and shown at the end (plt.show());
# run_all.py runs without windows (MPLBACKEND=Agg)
import matplotlib.pyplot as plt
import numpy as np
from pyxpp.builder import ModelBuilder

try:
    HERE = Path(__file__).resolve().parent
    NAME = Path(__file__).stem
except NameError:            # run as a notebook in Jupyter: the kernel starts in this folder
    HERE = Path.cwd()
    NAME = "2D-FLOWS_fitzhugh-nagumo"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
mb = (ModelBuilder("fitzhugh_nagumo")
      .comment("FitzHugh-Nagumo excitable membrane")
      .ode("v", "v-v^3/3-w+i")
      .ode("w", "eps*(v+a-b*w)")
      .parameter(i=0.2, a=0.7, b=0.8, eps=0.08)
      .initial(v=-1.2, w=-0.6)
      .option(total=200, dt=0.05))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. phase plane at i = 0.2
win = (-2.6, 2.6, -1, 2)
nc = m.nullclines(x="v", y="w", xlo=win[0], xhi=win[1], ylo=win[2], yhi=win[3], mesh=200)
X, Y, U, V = m.vector_field("v", "w", *win, grid=21)     # XPP's right-hand side on a grid
eq = m.equilibrium({"v": -1.2, "w": -0.6})               # Newton from a guess
print("\nequilibrium:", eq, "\n  eigenvalues:", np.round(eq.eigenvalues, 5))

fig, ax = plt.subplots(figsize=(6.5, 5))
S = np.hypot(U, V)
ax.quiver(X, Y, U / S, V / S, S, cmap="Greys", scale=35, width=0.003)
ax.plot(nc["V"][:, 0], nc["V"][:, 1], color="tab:orange", lw=2, label="v' = 0")
ax.plot(nc["W"][:, 0], nc["W"][:, 1], color="tab:green", lw=2, label="w' = 0")
traj_colors = ["tab:blue", "tab:purple", "tab:red", "tab:brown"]   # nullclines keep orange/green
for (v0, w0), col in zip([(-1.2, -0.6), (0.0, -0.5), (1.5, 1.5), (-2.0, 1.5)], traj_colors):
    tr = m.run(keep=False, v=v0, w=w0)                   # keep=False: don't change the model's ICs
    ax.plot(tr["v"], tr["w"], lw=1.3, color=col)
    ax.plot(v0, w0, "k.", ms=6)
ax.plot(eq.x["V"], eq.x["W"], "ko", ms=8, label="stable equilibrium")
ax.set(xlim=win[:2], ylim=win[2:], xlabel="v", ylabel="w", title="FitzHugh-Nagumo, i = 0.2")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "phase_plane.png", dpi=130)

# %% 3. excitability: kick v away from rest by increasing amounts
kicks = np.linspace(0.1, 1.2, 23)
peaks = [m.run(keep=False, total=60, v=eq.x["V"] + k, w=eq.x["W"])["v"].max() for k in kicks]
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(kicks, peaks, "o-")
ax.set(xlabel="kick in v", ylabel="largest v reached", title="All-or-none response")
fig.tight_layout()
fig.savefig(FIG / "excitability.png", dpi=130)
threshold = kicks[np.argmax(np.array(peaks) > 1.0)]
print(f"\nspike threshold: a kick of about {threshold:.2f} in v")

# %% 4. AUTO: equilibria in i, Hopf points, periodic orbits
m.set(i=0.0)
m.run(total=300)                 # settle on the rest state at i = 0
m.run_last(total=10)             # start from where that run ended (AUTO starts from the current state)
eqs = m.auto.run("i", start="steady", ds=0.01, dsmax=0.02, parmin=0, parmax=2, nmax=500)
hbs = eqs.special("HB")
print("\nAUTO, equilibria in i:")
for p in hbs:
    l1, om = m.auto.lyapunov(p)
    print(f"  Hopf at i = {p.x:.5f}: l1 = {l1:+.4f} ({'sub' if l1 > 0 else 'super'}critical), "
          f"period at onset {2 * np.pi / om:.3f}")

po = m.auto.run(grab=hbs[0].label, choice="periodic", ntst=60, ds=0.01, dsmax=0.05, nmax=300)
print(f"  {len(po)} periodic orbits from the first Hopf point;",
      "folds of orbits at i =", sorted({round(p.x, 4) for p in po.special("LP")}))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
xs = np.array([p.x for p in eqs])
vs = np.array([p.u0["v"] for p in eqs])
st = np.array([p.stable for p in eqs])
ax.plot(xs, np.where(st, vs, np.nan), "r-", lw=2.5, label="stable equilibria")
ax.plot(xs, np.where(~st, vs, np.nan), "k-", lw=1, label="unstable equilibria")
for f in ("umax", "umin"):
    ys = np.array([getattr(p, f)["v"] for p in po])
    pst = np.array([p.stable for p in po])
    ax.plot(np.array([p.x for p in po])[pst], ys[pst], "g.", ms=3)
    ax.plot(np.array([p.x for p in po])[~pst], ys[~pst], ".", color="tab:blue", ms=3)
ax.plot([], [], "g.", label="stable orbits (max/min v)")
ax.plot([], [], ".", color="tab:blue", label="unstable orbits")
ax.plot([p.x for p in hbs], [p.u0["v"] for p in hbs], "o", color="purple", ms=8, label="Hopf")
ax.set(xlim=(0, 2), xlabel="i", ylabel="v", title="Bifurcation diagram")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "bifurcation.png", dpi=130)

# %% 5. checks
print("\nchecks:")
check("equilibrium v at i = 0.2", eq.x["V"], -1.0692, 1e-3)
check("equilibrium is stable (1 = yes)", float(eq.stable), 1, 0)
# exact: Hopf where the trace vanishes, 1 - v^2 = eps*b, so v = -/+ c with c = sqrt(1 - eps*b),
# and i = v^3/3 - v + (v + a)/b on the equilibrium curve
a, b, eps = 0.7, 0.8, 0.08
c = np.sqrt(1 - eps * b)
i_hopf = sorted(s * (c**3 / 3 - c + c / b) + a / b for s in (-1, 1))
check("first Hopf point (exact)", hbs[0].x, i_hopf[0], 1e-4)
check("second Hopf point (exact)", hbs[1].x, i_hopf[1], 1e-4)
check("period at onset 2*pi/sqrt(det) (exact)", 2 * np.pi / m.auto.lyapunov(hbs[0])[1],
      2 * np.pi / np.sqrt(eps * (1 - b * (1 - c**2))), 1e-3)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
