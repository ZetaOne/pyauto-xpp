"""
Delayed logistic map: a Neimark-Sacker bifurcation
==================================================

x -> y, y -> a y (1 - x): a logistic map whose growth depends on the previous
generation too.  Its fixed point (1 - 1/a, 1 - 1/a) is stable up to a = 2, where
a complex pair of multipliers leaves the unit circle.  Instead of a period
doubling, an invariant closed curve is born (a Neimark-Sacker bifurcation, the
map version of a Hopf bifurcation), and orbits turn around it about once every
six iterates.

This script
  1. builds the map with ModelBuilder and prints the .ode file it makes
  2. computes the fixed point and its multipliers with XPP for several a
  3. follows the fixed point with AUTO and finds where the multipliers cross
     the unit circle
  4. draws orbits before and after a = 2, and measures the size of the
     invariant curve and the rotation number on it
     (at a = 2.2 the orbit has locked onto a cycle of period 7)
  5. checks the results against exact values derived from the Jacobian

Run:   python 13_MAPS_delayed-logistic-map.py
Figures go to figures/13_MAPS_delayed-logistic-map/.

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
    NAME = "13_MAPS_delayed-logistic-map"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
mb = (ModelBuilder("delayed_logistic_map")
      .comment("delayed logistic map")
      .map("x", "y")
      .map("y", "a*y*(1-x)")
      .parameter(a=2.1)
      .initial(x=0.6, y=0.6)
      .option(total=4000, meth="discrete"))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. the fixed point and its multipliers
# exact: the Jacobian [[0, 1], [-a y, a (1 - x)]] at x = y = 1 - 1/a is [[0, 1], [-(a-1), 1]]:
# trace 1, determinant a - 1.  For a > 5/4 the multipliers are a complex pair with
# |lambda|^2 = det = a - 1, so |lambda| = sqrt(a - 1) passes 1 at a = 2, where
# cos(arg lambda) = trace / (2 |lambda|) = 1/2, i.e. arg lambda = pi/3: one sixth of a turn.
A = np.linspace(1.3, 2.3, 21)
mods = []
for a in A:
    m.pars["a"] = a
    e = m.equilibrium({"x": 1 - 1 / a, "y": 1 - 1 / a})
    mods.append(np.abs(e.eigenvalues).max())
m.pars["a"] = 2.1
e21 = m.equilibrium({"x": 0.5, "y": 0.5})
print("\na = 2.1: fixed point", e21.x, "multipliers", np.round(e21.eigenvalues, 6))

# %% 3. AUTO: where the fixed point loses stability
m.run(a=1.9, x=0.4, y=0.4, total=2000)
m.run_last()                     # AUTO starts from the fixed point reached at a = 1.9
d = m.auto.run("a", start="steady", ds=0.01, dsmax=0.02, parmin=1.5, parmax=2.3, nmax=200, epss=1e-7)
ns = d.special("HB")[0]          # for a map, HB = multipliers leave the unit circle
mu = ns.eigenvalues[np.argmax(ns.eigenvalues.imag)]
print(f"AUTO: Neimark-Sacker at a = {ns.x:.7f}, multipliers {mu:.6f} and its conjugate")

# %% 4. orbits, size of the invariant curve, rotation number
def curve(a, n=4000, trans=30000):
    tr = m.run(keep=False, a=a, x=0.55, y=0.55, total=trans + n, trans=trans)
    return tr["x"], tr["y"]


def rotation(a):
    """turns per iterate around the fixed point (the map turns clockwise in (x, y))"""
    x, y = curve(a)
    th = np.unwrap(np.arctan2(y - (1 - 1 / a), x - (1 - 1 / a)))
    return -(th[-1] - th[0]) / (2 * np.pi * (len(th) - 1))


da = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.16])
size = np.array([np.ptp(curve(2 + s)[0]) / 2 for s in da])     # half the width in x
slope = np.polyfit(np.log(da[:3]), np.log(size[:3]), 1)[0]
rot = np.array([rotation(2 + s) for s in da[:3]])
rot0 = np.polyval(np.polyfit(da[:3], rot, 1), 0)                # extrapolate to a = 2
print(f"curve size ~ (a-2)^{slope:.4f}; rotation numbers {np.round(rot, 5)} -> {rot0:.5f} at a = 2")

fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
tr = m.run(keep=False, a=1.95, x=0.3, y=0.3, total=300)
axs[0].plot(tr["x"], tr["y"], ".", ms=3, color="tab:gray", label="a = 1.95: spirals in")
for a, col, lab in zip((2.02, 2.1, 2.2), ("tab:blue", "tab:green", "tab:red"), ("", "", ": locked to a 7-cycle")):
    axs[0].plot(*curve(a, n=1500), ".", ms=1.5 if a < 2.2 else 5, color=col, label=f"a = {a}{lab}")
axs[0].plot(0.5, 0.5, "k+", ms=10)
axs[0].set(xlabel="x", ylabel="y", title="Orbits and invariant curves")
axs[0].legend(fontsize=8, markerscale=4)
Af = np.linspace(1.3, 2.3, 200)
axs[1].plot(A, mods, "o", label="XPP")
axs[1].plot(Af, np.sqrt(Af - 1), "k-", lw=1, label="sqrt(a - 1)")
axs[1].axhline(1, color="tab:red", lw=0.8)
axs[1].plot(ns.x, 1, "s", color="purple", ms=8, label="AUTO's crossing")
axs[1].set(xlabel="a", ylabel="|multiplier|", title="Multipliers of the fixed point")
axs[1].legend(fontsize=8)
axs[2].loglog(da, size, "o", label="XPP")
axs[2].loglog(da, size[0] * np.sqrt(da / da[0]), "k-", lw=1, label="slope 1/2")
axs[2].set(xlabel="a - 2", ylabel="half width of the curve", title="Curve grows like sqrt(a - 2)")
axs[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "neimark_sacker.png", dpi=130)

# %% 5. checks
print("\nchecks:")
check("fixed point x at a = 2.1, 1 - 1/a (exact)", e21.x["X"], 1 - 1 / 2.1, 1e-9)
check("largest error in |multiplier| vs sqrt(a - 1) (exact 0)", np.max(np.abs(mods - np.sqrt(A - 1))), 0, 1e-9)
check("AUTO: multipliers cross the unit circle at a = 2 (exact)", ns.x, 2.0, 1e-6)
check("AUTO: |multiplier| there (exact 1)", abs(mu), 1.0, 1e-6)
check("AUTO: arg multiplier there, pi/3 (exact)", np.angle(mu), np.pi / 3, 1e-6)
check("rotation number on the curve extrapolated to a = 2, 1/6 (exact)", rot0, 1 / 6, 5e-4)
check("size of the curve ~ (a - 2)^p, p (normal form 1/2)", slope, 0.5, 0.02)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
