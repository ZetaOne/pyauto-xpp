"""
Two neurons with delayed coupling: a Hopf curve in (gain, delay)
================================================================

    x' = -x + a tanh(y(t-tau))
    y' = -y - a tanh(x(t-tau))

Two model neurons excite and inhibit each other through connections with a
transmission delay tau (Marcus and Westervelt studied such delayed networks).
The rest state (0, 0) has the characteristic equation
(lambda + 1)^2 + a^2 exp(-2 lambda tau) = 0; for tau = 1.3 a pair of roots
crosses the imaginary axis at a = 1.235459 and another at a = 2.870196, and a
stable oscillation appears at the first crossing.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. follows the rest state in a with AUTO and finds the two Hopf points,
     checked against their exact values
  3. follows both Hopf points in (a, tau) and checks every point of the curves
     against the characteristic equation
  4. continues the periodic orbits from the first Hopf point and compares them
     with simulations

Run:   python 35_DELAYS_marcus-westervelt.py
Figures go to figures/35_DELAYS_marcus-westervelt/.

Model from the delay-equation examples accompanying XPPAUT 8.2.
Every number is computed by XPPAUT 8.2 and its AUTO through pyxpp.
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
    NAME = "35_DELAYS_marcus-westervelt"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.7g} (expected {expected:.7g})")


# %% 1. the model
# The history (t < 0) is XPP's x(0)= formula: here the parameters x0, y0.  A run from a
# constant history sets both x0 and x (and y0, y).
mb = (ModelBuilder("marcus_westervelt")
      .comment("two neurons with delayed antisymmetric coupling")
      .ode("x", "-x+a*tanh(delay(y,tau))")
      .ode("y", "-y-a*tanh(delay(x,tau))")
      .parameter(a=0.5, tau=1.3, x0=0, y0=0)
      .initial_value("x", "x0")
      .initial_value("y", "y0")
      .initial(x=0, y=0)
      .option(delay=3, meth="rk4", total=100, dt=0.01, bound=1000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")
m.auto.parameters("a", "tau")


def hopf_exact(tau, k):
    """lambda = i w: (1 + i w) = +-i a exp(-i w tau) gives a^2 = 1 + w^2 and
    atan(w) + w tau = pi/2 + k pi; solved for w by bisection"""
    lo, hi = 0.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.arctan(mid) + mid * tau < np.pi / 2 + k * np.pi else (lo, mid)
    w = (lo + hi) / 2
    return np.sqrt(1 + w * w), w


# %% 2. the rest state in a, tau = 1.3
m.run(total=1)
m.run_last(total=1)
eqs = m.auto.run("a", start="steady", ds=0.01, dsmax=0.002, parmin=0.3, parmax=3.5, nmax=3000)
hbs = eqs.special("HB")
for p in hbs:
    print(f"\nHopf at a = {p.x:.6f}, omega = {2 * np.pi / p.period:.6f}")

# %% 3. Hopf curves in (a, tau)
curves = []
for h in hbs:
    for s in (1, -1):
        # epsl/epsu: AUTO's Newton tolerances (default 1e-4, which leaves errors of about 1e-3)
        c = m.auto.run(grab=h.label, choice="two param", ds=s * 0.01, dsmax=0.05, nmax=200,
                       parmin=0.3, parmax=3.5, epsl=1e-8, epsu=1e-8)
        curves.append([p for p in c if p.par["tau"] <= 6])
res, wres = [], []
for c in curves:
    for p in c:
        a, tau, w = p.x, p.par["tau"], 2 * np.pi / p.period
        res.append(abs((1j * w + 1)**2 + a * a * np.exp(-2j * w * tau)) / (1 + w * w))
        wres.append(abs(w - np.sqrt(a * a - 1)))
npts = sum(len(c) for c in curves)
print(f"Hopf curves: {npts} points")
m.set(a=0.5, tau=1.3)                        # AUTO leaves the last point's parameters behind

# %% 4. periodic orbits from the first Hopf point, and simulations
po = m.auto.run(grab=hbs[0].label, choice="periodic", ds=0.01, dsmax=0.05, nmax=200, parmax=3.0)
pa = np.array([p.x for p in po])
pmax = np.array([p.umax["x"] for p in po])
pper = np.array([p.period for p in po])
print(f"{len(po)} periodic orbits, stable at {sum(p.stable for p in po)}")
a_sim = np.array([1.4, 1.8, 2.4])
sims = []
for a in a_sim:
    tr = m.run(keep=False, a=a, tau=1.3, x=0.1, x0=0.1, y=0, y0=0, total=300, trans=200)
    t, x = tr.t, tr["x"]
    up = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    tc = t[up] - x[up] / (x[up + 1] - x[up]) * (t[up + 1] - t[up])
    sims.append((x.max(), np.diff(tc).mean()))
sims = np.array(sims)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
tt = np.linspace(0.01, 6, 400)
for k in range(4):                           # exact curves, from hopf_exact
    pts = np.array([hopf_exact(t_, k) for t_ in tt])
    a2.plot(pts[:, 0], tt, color="0.8", lw=6, label="exact (characteristic equation)" if k == 0 else None)
for i, c in enumerate(curves):
    a2.plot([p.x for p in c], [p.par["tau"] for p in c], "k-", lw=1.3,
            label="AUTO Hopf curves" if i == 0 else None)
a2.plot([p.x for p in hbs], [1.3, 1.3], "o", color="purple", ms=7, label="Hopf points, tau = 1.3")
a2.set(xlim=(0.9, 3.5), ylim=(0, 6), xlabel="a", ylabel="tau", title="Hopf curves of the rest state")
a2.legend(fontsize=8, loc="upper right")
st = np.array([p.stable for p in eqs])
xs = np.array([p.x for p in eqs])
a1.plot(xs, np.where(st, 0, np.nan), "r-", lw=2.5, label="stable rest state")
a1.plot(xs, np.where(~st, 0, np.nan), "k-", lw=1, label="unstable rest state")
a1.plot(pa, pmax, "g-", lw=2, label="AUTO orbits: max x")
a1.plot(a_sim, sims[:, 0], "ko", mfc="none", ms=8, label="simulations")
a1.plot([p.x for p in hbs], [0, 0], "o", color="purple", ms=7, label="Hopf")
a1.set(xlim=(0.5, 3.0), xlabel="a", ylabel="x", title="tau = 1.3")
a1.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "bifurcation.png", dpi=130)

# %% 5. checks
print("\nchecks:")
for p, (k, a_ref, w_ref) in zip(hbs, ((0, 1.235459, 0.725505), (1, 2.870196, 2.690358))):
    a_ex, w_ex = hopf_exact(1.3, k)
    check(f"Hopf at a = {a_ref} (exact {a_ex:.7f})", p.x, a_ex, 2e-5)
    check(f"  omega there ({w_ref})", 2 * np.pi / p.period, w_ex, 2e-5)
check(f"Hopf curves: largest |(iw+1)^2 + a^2 exp(-2iw tau)| / (1+w^2) on {npts} points",
      max(res), 0, 1e-4)
check("Hopf curves: largest |w - sqrt(a^2 - 1)|", max(wres), 0, 1e-4)
check("orbits from the first Hopf point are stable (1 = yes)", float(all(p.stable for p in po)), 1, 0)
for a, (smax, sper) in zip(a_sim, sims):
    check(f"a = {a}: max x, AUTO vs simulation", np.interp(a, pa, pmax), smax, 2e-3)
    check(f"a = {a}: period, AUTO vs simulation", np.interp(a, pa, pper), sper, 2e-3)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
