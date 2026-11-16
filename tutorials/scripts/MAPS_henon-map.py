"""
The Henon map: a strange attractor
==================================

x -> 1 - a x^2 + y, y -> b x.  The map stretches and folds the plane while
shrinking areas by the constant factor |b|, so orbits settle on an attractor of
zero area.  At a = 1.4, b = 0.3 that attractor is a fractal (Henon 1976); as a
grows from 0 it is reached through a cascade of period doublings.

This script
  1. builds the map with ModelBuilder and prints the .ode file it makes
  2. draws the attractor with the two fixed points of the map
  3. computes the fixed points and their multipliers with XPP
  4. gets the largest Liapunov exponent from XPP, and both exponents from the
     orbit; their sum is ln b because the Jacobian determinant is -b
  5. draws the orbit diagram in a with the first period doubling
  6. checks the results against exact and literature values

Run:   python MAPS_henon-map.py
Figures go to figures/MAPS_henon-map/.

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


# %% 1. the model
a, b = 1.4, 0.3
mb = (ModelBuilder("henon_map")
      .comment("the Henon map")
      .map("x", "1-a*x^2+y")
      .map("y", "b*x")
      .parameter(a=a, b=b)
      .initial(x=0, y=0)
      .option(total=100000, meth="discrete", bound=100))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. fixed points
# exact: x = 1 - a x^2 + b x, so a x^2 + (1-b) x - 1 = 0:
#   x = (-(1-b) +/- sqrt((1-b)^2 + 4a)) / (2a),  y = b x
xs = [(-(1 - b) + s * np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a) for s in (1, -1)]
# XPP's defaults NEWT_TOL = JAC_EPS = 1e-3 give the fixed point to ~7 digits and the
# multipliers to ~3 (they come from a finite-difference Jacobian with step JAC_EPS*|x|)
m.set(newt_tol=1e-6, jac_eps=1e-7)
fps = [m.equilibrium({"x": x, "y": b * x + 0.01}) for x in xs]
for fp in fps:
    print(f"\nfixed point ({fp.x['X']:.8f}, {fp.x['Y']:.8f}), multipliers {np.round(fp.eigenvalues.real, 6)}")

# %% 3. the attractor and XPP's Liapunov exponent
tr = m.run()                              # 100000 iterates from (0, 0)
x, y = tr["x"][100:], tr["y"][100:]
# XPP's Liapunov perturbs each stored point by JAC_EPS and steps it once; the stored orbit
# has single precision (~1e-7), so JAC_EPS must stay well above that: back to 1e-3 here
m.set(jac_eps=1e-3)
m.buttons.numerics.stochast.liapunov(0)  # Numerics > stocHast > Liapunov, on the run just made
msg = m.transcript.strip().splitlines()[-1]
lam_xpp = float(msg.split()[-1])
print("XPP:", msg.split("message")[-1].strip())

# both exponents from the orbit: carry two tangent vectors along with the Jacobian
# J = [[-2 a x, 1], [b, 0]] and re-orthonormalise them every step (QR)
Q, sums = np.eye(2), np.zeros(2)
for xn in x:
    Q, R = np.linalg.qr(np.array([[-2 * a * xn, 1.0], [b, 0.0]]) @ Q)
    sums += np.log(np.abs(np.diag(R)))
lam1, lam2 = sums / len(x)
print(f"from the orbit: exponents {lam1:.5f}, {lam2:.5f}; sum {lam1 + lam2:.6f}, ln b = {np.log(b):.6f}")
print(f"Kaplan-Yorke dimension 1 + lam1/|lam2| = {1 + lam1 / abs(lam2):.4f}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, lim in zip(axs, [(-1.5, 1.5, -0.45, 0.45), (0.55, 0.75, 0.15, 0.22)]):
    ax.plot(x, y, ",k")
    ax.plot([f.x["X"] for f in fps], [f.x["Y"] for f in fps], "o", mfc="none", mec="tab:red", ms=8,
            label="fixed points (saddles)")
    ax.set(xlim=lim[:2], ylim=lim[2:], xlabel="x", ylabel="y")
axs[0].set_title(f"Henon attractor, a = {a}, b = {b}")
axs[0].legend(fontsize=8, loc="upper left")
axs[1].set_title("Zoom near a fixed point: layers within layers")
fig.tight_layout()
fig.savefig(FIG / "attractor.png", dpi=150)

# %% 4. orbit diagram in a (b = 0.3)
# the fixed point with x > 0 has multipliers solving mu^2 + 2 a x mu - b = 0; mu = -1 gives
# 1 - 2 a x - b = 0, i.e. x = (1-b)/(2a), and on the fixed point curve a = 3(1-b)^2/4
a_pd = 3 * (1 - b) ** 2 / 4
A = np.linspace(0, 1.42, 1400)
P = np.repeat(A, 200)
X = np.concatenate([m.run(keep=False, a=aa, x=0, y=0, total=700, trans=500)["x"][1:] for aa in A])
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(P, X, s=0.3, c="k", marker=".", lw=0)
ax.axvline(a_pd, color="tab:blue", ls=":", lw=1)
ax.text(a_pd + 0.01, -1.2, "first doubling\na = 3(1-b)^2/4", fontsize=8, color="tab:blue")
ax.set(xlim=(0, 1.42), xlabel="a", ylabel="x", title="Orbit diagram of the Henon map, b = 0.3")
fig.tight_layout()
fig.savefig(FIG / "orbit_diagram.png", dpi=150)

m.set(a=a_pd, jac_eps=1e-7)
mu_pd = m.equilibrium({"x": (1 - b) / (2 * a_pd), "y": b * (1 - b) / (2 * a_pd)}).eigenvalues.real.min()
m.set(a=a)

# %% 5. checks
print("\nchecks:")
check("fixed point x+ (exact)", fps[0].x["X"], xs[0], 1e-8)
check("fixed point x- (exact)", fps[1].x["X"], xs[1], 1e-8)
check("product of multipliers = det J = -b (exact)", np.prod(fps[0].eigenvalues).real, -b, 1e-8)
check("multiplier -1 at the first doubling a = 3(1-b)^2/4 (exact)", mu_pd, -1.0, 1e-6)
check("XPP's largest Liapunov exponent (literature 0.419)", lam_xpp, 0.419, 0.01)
check("largest exponent from the orbit (literature 0.419)", lam1, 0.419, 0.01)
check("sum of the exponents = ln b (exact)", lam1 + lam2, np.log(b), 1e-9)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
