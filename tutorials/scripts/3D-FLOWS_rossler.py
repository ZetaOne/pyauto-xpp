"""
The Rossler system: period doubling to chaos
============================================

A simpler chaotic flow than Lorenz's, with one nonlinear term (z*x).  The orbit
winds around an unstable spiral near the origin and is now and then thrown up in
z and folded back.  As c grows the periodic orbit doubles its period again and
again, and at c = 5.7 the flow is chaotic.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. finds the equilibria: the inner one with XPP's Newton solver, both with
     AUTO, which follows them in c to the fold where they are born
  3. draws orbits at four values of c: period 1, 2, 4 and chaos
  4. sweeps c and plots the maxima of x after transients (the bifurcation
     diagram of the period-doubling cascade)
  5. locates the first two period doublings from the splitting of the maxima
  6. checks the results against exact values and the literature

Run:   python 3D-FLOWS_rossler.py
Figures go to figures/3D-FLOWS_rossler/.

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

HERE = Path(__file__).resolve().parent
NAME = Path(__file__).stem
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


def maxima(x):
    k = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    return x[k]


# %% 1. the model
a, b, c = 0.2, 0.2, 5.7
mb = (ModelBuilder("rossler")
      .comment("Rossler system")
      .ode("x", "-y-z")
      .ode("y", "x+a*y")
      .ode("z", "b+z*(x-c)")
      .parameter(a=a, b=b, c=c)
      .initial(x=1, y=1, z=0)
      .option(total=300, dt=0.02, meth="rk4", bound=1000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. equilibria: Newton for the inner one, AUTO for both
inner = m.equilibrium({"x": 0, "y": 0, "z": 0})
print("\ninner equilibrium:", inner, "\n  eigenvalues:", np.round(inner.eigenvalues, 4))
# The outer equilibrium has one unstable real direction, and m.equilibrium
# reports it as not found: XPP's Newton step converges, but the extra step that
# computes the eigenvector of that direction gives up, and its error flag comes
# back as a failure.  Reach it with AUTO instead: start on the inner equilibrium, follow it down in c through
# the fold where the two equilibria are born, and up the outer branch to c = 5.7.
m.set(x=inner.x["X"], y=inner.x["Y"], z=inner.x["Z"])
eqs = m.auto.run("c", start="steady", ds=-0.02, dsmax=0.1, parmin=0.2, parmax=c, nmax=600)
fold = eqs.special("LP")[0]
outer = eqs[-1]                           # end of the outer branch (AUTO stops just past c = 5.7)
print(f"AUTO: fold at c = {fold.x:.6f}; outer equilibrium at c = {outer.x:.4f}:",
      {k: round(v, 5) for k, v in outer.u0.items()})
m.set(x=1, y=1, z=0, c=c)                 # back to the model's start and c

# %% 3. orbits at four values of c
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True, sharey=True)
for ax, cc, lab in zip(axs, (2.5, 3.5, 4.0, 5.7), ("period 1", "period 2", "period 4", "chaos")):
    tr = m.run(keep=False, c=cc, total=700, trans=500)
    ax.plot(tr["x"], tr["y"], lw=0.6)
    ax.set(title=f"c = {cc}: {lab}", xlabel="x")
axs[0].set_ylabel("y")
fig.tight_layout()
fig.savefig(FIG / "orbits.png", dpi=130)

tr = m.run(keep=False, total=500, trans=200)
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
ax.plot(tr["x"], tr["y"], tr["z"], lw=0.4)
ax.set(xlabel="x", ylabel="y", zlabel="z", title="Rossler attractor, c = 5.7")
fig.tight_layout()
fig.savefig(FIG / "attractor.png", dpi=130)

# %% 4. maxima of x against c
C, M = [], []
for cc in np.linspace(2, 6.5, 226):
    mx = maxima(m.run(keep=False, c=cc, total=800, trans=500)["x"])
    C += [cc] * len(mx)
    M += list(mx)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(C, M, "k.", ms=0.8)
ax.plot([p.x for p in eqs], [p.u0["x"] for p in eqs], color="tab:gray", lw=1,
        label="equilibria (unstable)")
ax.legend(fontsize=8, loc="lower right")
ax.set(xlim=(2, 6.5), xlabel="c", ylabel="maxima of x", title="Period doubling in the Rossler system")

# %% 5. where the period doubles
# Just past a period doubling the maxima alternate, and the gap between them grows
# like sqrt(c - c_pd).  So gap^2 is linear in c: fit and extrapolate to zero.
# For the first doubling compare maxima one apart, for the second two apart.
def gap(cc, lag):
    mx = maxima(m.run(keep=False, c=cc, total=3000, trans=2500)["x"])[-20:]
    return np.mean(np.abs(mx[lag:] - mx[:-lag]))


c_pd = []
for lag, cs in ((1, np.linspace(2.86, 2.98, 7)), (2, np.linspace(3.87, 3.95, 5))):
    g2 = np.array([gap(cc, lag)**2 for cc in cs])
    slope, icept = np.polyfit(cs, g2, 1)
    c_pd.append(-icept / slope)
for cp in c_pd:
    ax.axvline(cp, color="tab:red", lw=0.8, ls="--")
ax.text(c_pd[0], ax.get_ylim()[1], f" {c_pd[0]:.3f}", color="tab:red", va="top", fontsize=8)
ax.text(c_pd[1], ax.get_ylim()[1], f" {c_pd[1]:.3f}", color="tab:red", va="top", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "maxima_vs_c.png", dpi=130)
print(f"\nperiod doublings at c = {c_pd[0]:.4f} and {c_pd[1]:.4f}")

# %% 6. checks
print("\nchecks:")
# exact: x' = 0 gives z = -y, y' = 0 gives y = -x/a, so z = x/a, and z' = 0 gives
# x^2 - c x + a b = 0, x = (c -+ sqrt(c^2 - 4ab))/2; the two meet at c = 2 sqrt(ab)
def x_eq(cc, sign):
    return (cc + sign * np.sqrt(cc**2 - 4 * a * b)) / 2


check("inner equilibrium x (exact)", inner.x["X"], x_eq(c, -1), 1e-8)
check("inner equilibrium y = -x/a (exact)", inner.x["Y"], -x_eq(c, -1) / a, 1e-7)
check("inner equilibrium z = x/a (exact)", inner.x["Z"], x_eq(c, -1) / a, 1e-7)
check("outer equilibrium x (exact, at AUTO's c)", outer.u0["x"], x_eq(outer.x, 1), 1e-6)
check("outer equilibrium z = x/a (exact)", outer.u0["z"], outer.u0["x"] / a, 1e-6)
check("fold of equilibria at c = 2 sqrt(ab) (exact)", fold.x, 2 * np.sqrt(a * b), 1e-6)
# literature (a = b = 0.2): period doublings at c = 2.83 and 3.84
check("first period doubling (literature 2.83)", c_pd[0], 2.83, 0.02)
check("second period doubling (literature 3.84)", c_pd[1], 3.84, 0.02)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
