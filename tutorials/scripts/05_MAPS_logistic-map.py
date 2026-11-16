"""
The logistic map: period doubling to chaos
==========================================

x -> r x (1 - x), the simplest map with chaos.  The fixed point 1 - 1/r loses
stability at r = 3 when its multiplier r(1 - 2x) passes -1; a period-2 orbit
takes over, then period 4 at r = 1 + sqrt(6), 8, 16, ... accumulating near
r = 3.5699, after which most r give chaos, broken by periodic windows.

This script
  1. builds the map with ModelBuilder and prints the .ode file it makes
  2. draws the orbit diagram: the long-run iterates against r
  3. computes the Liapunov exponent from XPP's orbits, the mean of ln|r(1 - 2x)|,
     which touches 0 at every period doubling and is positive in chaos
  4. follows the fixed point with AUTO and finds where it loses stability
  5. checks the results against exact values: r = 3, multiplier -1, the
     multipliers of the fixed point and of the period-2 orbit, the onset of
     period 4 at 1 + sqrt(6), and the exponent ln 2 at r = 4

Run:   python 05_MAPS_logistic-map.py
Figures go to figures/05_MAPS_logistic-map/.

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
    NAME = "05_MAPS_logistic-map"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
mb = (ModelBuilder("logistic_map")
      .comment("the logistic map")
      .map("x", "r*x*(1-x)")
      .parameter(r=3.7)
      .initial(x=0.5)
      .option(total=300, meth="discrete"))           # discrete: x(n+1) = r x(n) (1 - x(n))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP


def orbit(r, n=400, trans=1000, x0=0.3):
    """n iterates after the first trans (tr.t is the iterate number, trans ... trans + n)"""
    return m.run(keep=False, total=trans + n, trans=trans, r=r, x=x0)["x"][1:]


def liapunov(r, n=4000, trans=2000):
    x = orbit(r, n, trans)
    return np.mean(np.log(np.abs(r * (1 - 2 * x))))


# %% 2. orbit diagram and 3. Liapunov exponent
R = np.linspace(2.5, 4, 1500)
lam = np.array([liapunov(r) for r in R])
r_pd = [3, 1 + np.sqrt(6), 3.544090, 3.569946]       # doublings 1, 2, 3 and the accumulation point

fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
P = np.repeat(R, 250)
X = np.concatenate([orbit(r, 250) for r in R])
axs[0].scatter(P, X, s=0.25, c="k", marker=".", lw=0)
axs[0].set(ylabel="x", ylim=(0, 1), title="Orbit diagram of x -> r x (1 - x)")
axs[1].plot(R, lam, "k-", lw=0.6)
axs[1].axhline(0, color="tab:red", lw=0.8)
for r in r_pd:
    for ax in axs:
        ax.axvline(r, color="tab:blue", ls=":", lw=0.8)
for r, lab in zip(r_pd, ["3", "1+\u221a6", "", "r\u221e"]):
    axs[1].text(r - 0.005, 0.55, lab, ha="right", fontsize=8, color="tab:blue")
axs[1].set(xlabel="r", ylabel="Liapunov exponent", ylim=(-1.6, 0.8), xlim=(2.5, 4))
fig.tight_layout()
fig.savefig(FIG / "orbit_diagram.png", dpi=150)

# zoom onto the second doubling: the exponent rises to 0 at r2 and falls again
Rz = np.linspace(3.43, 3.47, 401)
lamz = np.array([liapunov(r, n=4000, trans=40000) for r in Rz])
r2 = Rz[np.argmax(lamz)]
print(f"\nperiod 2 -> 4: exponent largest (closest to 0) at r = {r2:.4f}; 1 + sqrt(6) = {1 + np.sqrt(6):.4f}")

# %% 4. AUTO: the fixed point and its loss of stability
m.run(r=2.8)
m.run_last()                     # start AUTO from the fixed point reached at r = 2.8
d = m.auto.run("r", start="steady", ds=0.01, dsmax=0.02, parmin=2.5, parmax=4, nmax=200, epss=1e-7)
hb = d.special("HB")[0]          # for a map, HB marks a multiplier leaving the unit circle
mu = hb.eigenvalues[0]
print(f"AUTO: fixed point x = {hb.u0['X']:.6f} loses stability at r = {hb.x:.7f}, multiplier {mu.real:.6f}")

# XPP's own Liapunov exponent (Numerics > stocHast > Liapunov) at r = 4
m.run(r=4, x=0.3, total=20000)
m.buttons.numerics.stochast.liapunov(0)
msg = m.transcript.strip().splitlines()[-1]
lam_xpp = float(msg.split()[-1])
print("XPP at r = 4:", msg.split("message")[-1].strip())

# %% 5. checks
print("\nchecks:")
check("AUTO: fixed point loses stability at r = 3 (exact)", hb.x, 3.0, 1e-6)
check("AUTO: multiplier there is -1 (exact)", mu.real, -1.0, 1e-6)
check("AUTO: fixed point there, 1 - 1/r = 2/3 (exact)", hb.u0["X"], 2 / 3, 1e-6)
# XPP stores trajectories in single precision (about 7 digits), hence tolerances of 1e-6
# below r = 3 the orbit sits on the fixed point, multiplier r(1 - 2(1 - 1/r)) = 2 - r
check("exponent at r = 2.8 from the orbit, ln|2 - r| (exact)", liapunov(2.8), np.log(0.8), 1e-6)
# period-2 points are the roots of r^2 x^2 - r(r+1) x + (r+1) = 0, so the multiplier
# r^2 (1 - 2x1)(1 - 2x2) = r^2 - 2r(r+1) + 4(r+1) = 4 + 2r - r^2; it is -1 at r = 1 + sqrt(6)
x12 = orbit(3.3, 2)
check("period-2 multiplier at r = 3.3 from the orbit, 4 + 2r - r^2 (exact)",
      np.prod(3.3 * (1 - 2 * x12)), 4 + 2 * 3.3 - 3.3**2, 1e-6)
check("onset of period 4 from the exponent, 1 + sqrt(6) (exact; grid step 1e-4)", r2, 1 + np.sqrt(6), 1e-4)
# at r = 4, x = sin^2(theta) turns the map into theta -> 2 theta: exponent ln 2
check("exponent at r = 4 from the orbit, ln 2 (exact; sampling error ~0.01)", liapunov(4, n=200000), np.log(2), 0.01)
check("XPP's Liapunov exponent at r = 4, ln 2 (exact)", lam_xpp, np.log(2), 0.01)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
