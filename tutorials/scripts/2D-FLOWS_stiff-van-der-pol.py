"""
Stiff van der Pol: when to use a stiff solver
=============================================

The van der Pol relaxation oscillator in Lienard form, x' = mu(x - x^3/3 - y),
y' = x/mu.  With mu = 1000 the orbit creeps along the outer branches of the
cubic y = x - x^3/3 on a time scale mu and jumps between them on a time scale
1/mu: the two speeds differ by a factor of a million.  An explicit method must
take steps of about 1/mu just to stay stable; XPP's stiff (Rosenbrock) solver
and CVODE take steps as long as the slow motion allows.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. integrates the same orbit with stiff, cvode and rk4 (two step sizes) and
     compares rows stored, run time and the period each one gives
  3. plots the relaxation cycle on the cubic nullcline and x against time
  4. measures the period over five decades of mu (XPP's Initialconds >
     Periodic where it works, crossing times everywhere) and compares it with
     2*pi for small mu and with the relaxation asymptotics for large mu
  5. checks the results against the theory

Run:   python 2D-FLOWS_stiff-van-der-pol.py
Figures go to figures/2D-FLOWS_stiff-van-der-pol/.

Model from the examples of XPP Studio (Peter Ashwin and Bard Ermentrout).
Every number is computed by XPPAUT 8.1 and its AUTO through pyxpp.
"""
# %% setup
import time
from pathlib import Path

import matplotlib
# figures are saved to figures/ and shown at the end (plt.show());
# run_all.py runs without windows (MPLBACKEND=Agg)
import matplotlib.pyplot as plt
import numpy as np
from pyxpp import Run
from pyxpp.builder import ModelBuilder

try:
    HERE = Path(__file__).resolve().parent
    NAME = Path(__file__).stem
except NameError:            # run as a notebook in Jupyter: the kernel starts in this folder
    HERE = Path.cwd()
    NAME = "2D-FLOWS_stiff-van-der-pol"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


def period_from_crossings(tr):
    """mean time between upward crossings of y = 0 (on the slow right branch, so
    linear interpolation between stored points is accurate)"""
    t, y = tr.t, tr["y"]
    i = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    tc = t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i])
    return np.diff(tc).mean() if len(tc) > 1 else np.nan


# %% 1. the model
mb = (ModelBuilder("stiff_van_der_pol")
      .comment("van der Pol relaxation oscillator, stiff for large mu")
      .ode("x", "mu*(x-x^3/3-y)")
      .ode("y", "x/mu")
      .parameter(mu=1000)
      .initial(x=2, y=0.6667)
      .option(meth="stiff", total=4000, dt=20, toler=1e-6))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. solvers compared on five periods at mu = 1000
solvers = [("stiff, dt=1", Run(meth="stiff", dt=1, tol=1e-6)),
           ("cvode, dt=1", Run(meth="cvode", dt=1, tol=1e-11, atol=1e-11)),
           ("rk4, dt=0.001", Run(meth="rk4", dt=0.001, nout=1000)),
           ("rk4, dt=0.0005", Run(meth="rk4", dt=0.0005, nout=2000))]
runs, periods = {}, {}
print("\nmu = 1000, t = 0..8000:")
for label, r in solvers:
    t0 = time.time()
    tr = m.run(r.replace(total=8000))
    el = time.time() - t0
    runs[label], periods[label] = tr, period_from_crossings(tr)
    print(f"  {label:15s} steps {8000 / r.dt:9.0f}  rows {len(tr):5d}  {el:5.2f} s  "
          f"period {periods[label]:.4f}")
# rk4 is stable only for |h*lambda| < 2.79; at x = 2 the fast eigenvalue is mu(1-x^2) = -3000,
# so h = 0.001 sits just outside the limit near the turning points and the period comes out wrong.

# %% 3. the relaxation cycle
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
xc = np.linspace(-2.5, 2.5, 400)
a1.plot(xc, xc - xc**3 / 3, color="tab:orange", lw=2, label="x' = 0 (y = x - x³/3)")
a1.axvline(0, color="tab:green", lw=2, label="y' = 0")
late = runs["stiff, dt=1"].t > 1000                  # leave out the jump from the initial point
a1.plot(runs["stiff, dt=1"]["x"][late], runs["stiff, dt=1"]["y"][late], color="tab:blue", lw=1.5,
        label="orbit (stiff)")
a1.set(xlim=(-2.5, 2.5), ylim=(-1, 1.6), xlabel="x", ylabel="y", title="mu = 1000: phase plane")
a1.legend(fontsize=8, loc="upper right")
for label, col, ls in [("stiff, dt=1", "tab:blue", "-"), ("rk4, dt=0.001", "tab:red", "--")]:
    tr = runs[label]
    a2.plot(tr.t, tr["x"], color=col, ls=ls, lw=1.3, label=f"{label}: T = {periods[label]:.2f}")
a2.set(xlim=(0, 8000), ylim=(-2.2, 2.8), xlabel="t", ylabel="x", title="x against time: an explicit step slightly too long")
a2.legend(fontsize=8, loc="upper center", ncol=2)
fig.tight_layout()
fig.savefig(FIG / "relaxation_cycle.png", dpi=130)

# %% 4. period against mu
mus = np.logspace(-1, 3, 17)
T_cross, T_per = [], []
for mu in mus:
    Tg = 2 * np.pi if mu < 1 else max(2 * np.pi, 1.61 * mu)      # rough guess, sets the run length
    tr = m.run(keep=False, mu=mu, x=2, y=0.6667, meth="cvode", tol=1e-10, atol=1e-10,
               dt=Tg / 2000, total=8 * Tg, trans=2 * Tg)
    T_cross.append(period_from_crossings(tr))
    T_per.append(np.nan)
    if mu <= 200:                          # beyond this XPP's Periodic reports the orbit is too stiff
        m.run(mu=mu, x=2, y=0.6667, meth="cvode", tol=1e-10, atol=1e-10, dt=Tg / 1000, total=6 * Tg, trans=0)
        m.buttons.initialconds.periodic()
        T_per[-1] = m.numerics["total"]
m.set(mu=1000, x=2, y=0.6667, meth="stiff", dt=20, total=4000, tol=1e-6)
T_cross, T_per = np.array(T_cross), np.array(T_per)

ALPHA = 2.338107                           # first zero of the Airy function of -z
mm = np.logspace(-1, 3, 200)
fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.loglog(mm, 2 * np.pi * (1 + mm**2 / 16) * np.ones_like(mm), color="tab:green", ls="--",
          label="2π(1 + mu²/16)  (small mu)")
ax.loglog(mm[mm > 2], (3 - 2 * np.log(2)) * mm[mm > 2] + 3 * ALPHA * mm[mm > 2] ** (-1 / 3),
          color="tab:orange", ls="--", label="(3 - 2 ln 2) mu + 3α mu^(-1/3)  (large mu)")
ax.loglog(mus, T_cross, "o", color="tab:blue", ms=6, label="crossing times")
ax.loglog(mus, T_per, "x", color="k", ms=7, label="Initialconds > Periodic")
ax.set(xlabel="mu", ylabel="period T", ylim=(4, 3000), title="Period of the van der Pol cycle")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "period_vs_mu.png", dpi=130)
for mu, a, b in zip(mus, T_cross, T_per):
    print(f"  mu = {mu:8.3f}   T (crossings) = {a:10.4f}   T (Periodic) = {b:10.4f}")

# %% 5. checks
print("\nchecks:")
# Slow branch: dy/dt = x/mu and dy = (1 - x^2) dx, so dt = mu (1 - x^2)/x dx; from x = 2 to the
# knee x = 1 this takes mu (3/2 - ln 2).  Two branches: T0 = (3 - 2 ln 2) mu.  The next term,
# from passing the knee, is 3*alpha*mu^(-1/3) (alpha = 2.338107); the rest is O(ln(mu)/mu), ~0.007 here.
T_asym = (3 - 2 * np.log(2)) * 1000 + 3 * ALPHA * 1000 ** (-1 / 3)
check("period at mu = 1000, stiff vs asymptotics", periods["stiff, dt=1"], T_asym, 0.02)
check("cvode agrees with stiff (period)", periods["cvode, dt=1"], periods["stiff, dt=1"], 1e-3)
check("rk4 dt=0.0005 agrees with stiff (period)", periods["rk4, dt=0.0005"], periods["stiff, dt=1"], 1e-3)
# Poincare-Lindstedt: T = 2*pi (1 + mu^2/16 + O(mu^4)); at mu = 0.1 the O(mu^4) term is ~1e-6
check("period at mu = 0.1 vs 2*pi(1 + mu^2/16)", T_per[0], 2 * np.pi * (1 + 0.01 / 16), 1e-4)
check("period at mu = 1 (literature 6.663287)", T_per[np.isclose(mus, 1)][0], 6.663287, 1e-5)
check("Periodic and crossing times agree at mu = 100", T_per[np.isclose(mus, 100)][0], T_cross[np.isclose(mus, 100)][0], 1e-3)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
