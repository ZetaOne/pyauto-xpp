"""
Hutchinson's delayed logistic equation: oscillations born in a Hopf point
=========================================================================

    x' = r x(t) (1 - x(t-1))

A population whose growth is regulated by its size one generation ago.  The
equilibrium x = 1 is stable for small r; at r = pi/2 a pair of roots of the
characteristic equation  lambda = -r exp(-lambda)  crosses the imaginary axis
at lambda = i pi/2, and a periodic orbit of period 4 appears.  Its amplitude
grows quickly with r, and for large r the population spends long times near 0.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. follows the equilibrium x = 1 in r with AUTO and finds the Hopf point
  3. continues the periodic orbits from the Hopf point with XPP 8.2's
     periodic-orbit solver for delay equations (with Floquet multipliers)
  4. imports one computed orbit and compares it with a long simulation
  5. compares amplitude and period along the branch with simulations, and
     checks the Hopf point, the period 4 and the amplitude law near onset

Run:   python DELAYS_hutchinson-periodic-orbits.py
Figures go to figures/DELAYS_hutchinson-periodic-orbits/.

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
# delay=2: XPP keeps 2 time units of past values.  The history (x for t < 0) is XPP's x(0)=
# formula, here the parameter x0; "init x=1" sets only the value at t = 0.  So a run from a
# constant history v sets both x0=v and x=v.
mb = (ModelBuilder("hutchinson")
      .comment("Hutchinson delayed logistic equation")
      .ode("x", "r*x*(1-delay(x,1))")
      .parameter(r=1.0, x0=1.0)
      .initial_value("x", "x0")
      .initial(x=1.0)
      .option(delay=2, meth="rk4", total=50, dt=0.01, maxstor=100000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")


def simulate(r, total=400, keep_last=60):
    """long run from the constant history x = 1.2: max, min and period of the last part"""
    tr = m.run(keep=False, r=r, x=1.2, x0=1.2, total=total, trans=total - keep_last)
    t, x = tr.t, tr["x"]
    up = np.where((x[:-1] < 1) & (x[1:] >= 1))[0]          # upward crossings of x = 1
    tc = t[up] + (1 - x[up]) / (x[up + 1] - x[up]) * (t[up + 1] - t[up])
    return x.max(), x.min(), np.diff(tc).mean(), tr


# %% 2. AUTO: the equilibrium x = 1 in r
m.run(total=20)                 # x = 1 at r = 1: already the equilibrium
m.run_last(total=1)             # AUTO starts from the current state
eqs = m.auto.run("r", start="steady", ds=0.01, dsmax=0.05, parmin=0.5, parmax=3, nmax=300)
hb = eqs.special("HB")[0]
print(f"\nHopf point at r = {hb.x:.7f}, period {hb.period:.6f}; "
      f"rightmost root there {hb.eigenvalues[0]:.5f}")

# %% 3. periodic orbits from the Hopf point
po = m.auto.run(grab=hb.label, choice="periodic", ds=0.01, dsmax=0.05, nmax=300, npr=5, parmax=3)
# npr=5: every 5th orbit is labeled; only labeled orbits are kept for Import and Extend
rs = np.array([p.x for p in po])
xmax = np.array([p.umax["x"] for p in po])
xmin = np.array([p.umin["x"] for p in po])
per = np.array([p.period for p in po])
stable = np.array([p.stable for p in po])
mult = np.array([abs(p.multipliers[0]) for p in po])   # largest nontrivial Floquet multiplier
print(f"{len(po)} periodic orbits for r in [{rs[0]:.4f}, {rs[-1]:.3f}]; "
      f"stable at {stable.sum()} of them; largest |multiplier| {mult.max():.3f}")

# %% 4. one orbit from AUTO next to a simulation
lab = min(po.labeled(), key=lambda p: abs(p.x - 2.0))  # the labeled orbit nearest r = 2
m.auto.grab(lab.label)
m.auto.buttons.file.import_orbit()                     # File > Import orbit: the orbit into XPP's data
orb = m.data
smax, smin, sper, tr = simulate(lab.x)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
i0 = np.argmax(tr["x"])                                 # start the simulation at a peak
ts = tr.t[i0:] - tr.t[i0]
ax.plot(ts, tr["x"][i0:], color="0.6", lw=4, label="simulation")
j0 = np.argmax(orb["x"])                                # the orbit from its peak, drawn twice
to = np.r_[orb.t[j0:] - orb.t[j0], orb.t[:j0] + orb.t[-1] - orb.t[j0]]
xo = np.r_[orb["x"][j0:], orb["x"][:j0]]
ax.plot(np.r_[to, to + lab.period], np.r_[xo, xo], "k--", lw=1.5, label="AUTO orbit")
ax.set(xlim=(0, 2 * lab.period), xlabel="t (from a peak)", ylabel="x",
       title=f"Periodic orbit at r = {lab.x:.4f} (period {lab.period:.4f})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "orbit.png", dpi=130)

# %% 5. amplitude and period along the branch, against simulations
r_sim = np.array([1.6, 1.8, 2.0, 2.3, 2.6, 2.9])
sims = np.array([simulate(r)[:3] for r in r_sim])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot([0.5, hb.x], [1, 1], "r-", lw=2.5, label="stable equilibrium")
a1.plot([hb.x, 3], [1, 1], "k-", lw=1, label="unstable equilibrium")
a1.plot(rs, xmax, "g-", lw=2, label="AUTO orbits: max and min of x")
a1.plot(rs, xmin, "g-", lw=2)
a1.plot(r_sim, sims[:, 0], "ko", mfc="none", ms=7, label="simulations")
a1.plot(r_sim, sims[:, 1], "ko", mfc="none", ms=7)
a1.plot(hb.x, 1, "o", color="purple", ms=8, label="Hopf, r = pi/2")
a1.set(xlim=(0.5, 3), xlabel="r", ylabel="x", title="Bifurcation diagram")
a1.legend(fontsize=8, loc="upper left")
a2.plot(rs, per, "g-", lw=2, label="AUTO")
a2.plot(r_sim, sims[:, 2], "ko", mfc="none", ms=7, label="simulations")
a2.plot(hb.x, 4, "o", color="purple", ms=8)
a2.set(xlabel="r", ylabel="period", title="Period of the orbit")
a2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "bifurcation.png", dpi=130)

print("\n     r    max x: AUTO   sim     min x: AUTO   sim      period: AUTO   sim")
for r, (sx, sn, sp) in zip(r_sim, sims):
    print(f"  {r:4.2f}   {np.interp(r, rs, xmax):10.5f} {sx:8.5f}   {np.interp(r, rs, xmin):10.5f} "
          f"{sn:8.5f}   {np.interp(r, rs, per):10.5f} {sp:8.5f}")

# %% 6. checks
print("\nchecks:")
# lambda = i w in lambda = -r exp(-lambda):  r cos w = 0, w = r sin w  ->  w = r = pi/2, period 4
check("Hopf point r = pi/2 (exact)", hb.x, np.pi / 2, 1e-5)
check("period at the Hopf point = 4 (exact)", hb.period, 4.0, 1e-4)
check("first orbit of the branch has period near 4", per[0], 4.0, 1e-3)
# with y = x - 1 this is Wright's equation y' = -r y(t-1) (1 + y); its Hopf orbit is
# y = A cos(pi t/2) with A^2 = 40 (r - pi/2) / (3 pi - 2)  (supercritical)
amp0 = (xmax[0] - xmin[0]) / 2
check("amplitude near onset / Wright's law", amp0 / np.sqrt(40 * (rs[0] - np.pi / 2) / (3 * np.pi - 2)),
      1.0, 1e-2)
check("all orbits stable (Floquet multipliers inside the unit circle; 1 = yes)",
      float(stable.all()), 1, 0)
check("imported orbit: max x equals AUTO's", orb["x"].max(), lab.umax["x"], 1e-4)
check(f"max x at r = {lab.x:.3f}: AUTO vs simulation", lab.umax["x"], smax, 2e-3)
check(f"period at r = {lab.x:.3f}: AUTO vs simulation", lab.period, sper, 2e-3)
err_max = np.max(np.abs(np.interp(r_sim, rs, xmax) - sims[:, 0]) / sims[:, 0])
err_per = np.max(np.abs(np.interp(r_sim, rs, per) - sims[:, 2]) / sims[:, 2])
check("largest relative difference of max x, AUTO vs 6 simulations", err_max, 0, 5e-3)
check("largest relative difference of the period, AUTO vs 6 simulations", err_per, 0, 5e-3)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
