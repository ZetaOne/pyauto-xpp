"""
A cusp with a delay: folds of equilibria and delay-induced Hopf points
======================================================================

    x' = a + b x(t-1) - x(t-1)^3

Without the delay this is the normal form of a cusp: three equilibria between
two folds, the outer two stable.  The delay does not move the equilibria (they
still solve a + b x - x^3 = 0) or the folds, but it changes their stability:
the linearization u' = c u(t-1), c = b - 3 x^2, loses stability when
c = -pi/2 (roots +-i pi/2, period 4) and a second pair crosses at c = -5 pi/2
(roots +-5i pi/2, period 0.8).  So the outer branches are stable only close
to the folds.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. follows the equilibria in a with AUTO (b = 3): two folds and four Hopf
     points, checked against their exact values
  3. follows a fold in (a, b) and checks it against a = (2b/3) sqrt(b/3)
  4. continues the periodic orbits born at the Hopf points: the period-4 orbit
     grows until its period diverges (it ends on a loop through the middle
     saddle); the period-0.8 orbits are unstable
  5. checks stability with simulations on both sides of a Hopf point

Run:   python DELAYS_cusp-delay.py
Figures go to figures/DELAYS_cusp-delay/.

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
    NAME = "DELAYS_cusp-delay"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.7g} (expected {expected:.7g})")


# %% 1. the model
mb = (ModelBuilder("cusp_delay")
      .comment("cusp normal form with a delayed argument")
      .ode("x", "a+b*delay(x,1)-delay(x,1)^3")
      .parameter(a=-3.0, b=3.0, x0=-2.1)
      .initial_value("x", "x0")      # x(0)=x0: the history x(t) = x0 for t <= 0 (see below)
      .option(delay=1, meth="rk4", total=50, dt=0.01, bound=1000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")
m.auto.parameters("a", "b")
# The history of a delay equation (x for -1 <= t <= 0) is XPP's x(0)= formula, here the
# parameter x0.  "init x=..." (ModelBuilder.initial) would set only x at t = 0 and leave the
# history at 0.  So a run from a constant history v needs both x0=v and x=v.

# %% 2. equilibria in a
# at a = -3 the only equilibrium (x = -2.104) is unstable, so a simulation would not find it:
# Newton does, and AUTO starts from the initial conditions
eq = m.equilibrium({"x": -2.1})
m.set(x=eq.x["X"], x0=eq.x["X"])
print(f"\nequilibrium at a = -3: x = {eq.x['X']:.6f}, rightmost root {eq.eigenvalues[0]:.4f}")
eqs = m.auto.run("a", start="steady", ds=0.01, dsmax=0.005, parmin=-3.5, parmax=3, nmax=3000)
lps, hbs = eqs.special("LP"), eqs.special("HB")
for p in lps + hbs:
    extra = f", period {p.period:.5f}" if p.type == "HB" else ""
    print(f"  {p.type} at a = {p.x:+.7f}, x = {p.u0['x']:+.6f}{extra}")

# %% 3. the fold in (a, b)
m.auto.axes(plot="two_par", param="a", param2="b", xmin=-3.5, xmax=3.5, ymin=-0.5, ymax=5)
# epsl/epsu: AUTO's Newton tolerances; the default 1e-4 leaves an error of 2e-4 on this curve
fold = [m.auto.run(grab=lps[0].label, ds=s * 0.01, dsmax=0.05, nmax=150, parmin=-3.5, parmax=3.5,
                   epsl=1e-8, epsu=1e-8) for s in (1, -1)]
fold_pts = [p for c in fold for p in c]
fa = np.array([p.x for p in fold_pts])
fb = np.array([p.par["b"] for p in fold_pts])
fx = np.array([p.u0["x"] for p in fold_pts])
m.set(a=-3.0, b=3.0)                       # AUTO leaves the last point's parameters behind

# %% 4. periodic orbits from the Hopf points
orbits = {}
for h in hbs:
    # the period-4 orbit's period diverges near a = -1.566 (+1.566 for its mirror): AUTO's mesh
    # then grows to thousands of unknowns and each step gets slow, so stop after 45 steps
    n = 45 if h.period > 2 else 60
    orbits[h.label] = m.auto.run(grab=h.label, choice="periodic", ds=0.01, dsmax=0.05, nmax=n,
                                 parmin=-3.5, parmax=3)
    o = orbits[h.label]
    print(f"  orbits from the Hopf point at a = {h.x:+.4f}: {len(o)} orbits up to a = {o[-1].x:+.4f}, "
          f"period {o[0].period:.3f} -> {o[-1].period:.3f}, stable at {sum(p.stable for p in o)}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2))
xs = np.array([p.x for p in eqs])
us = np.array([p.u0["x"] for p in eqs])
st = np.array([p.stable for p in eqs])
a1.plot(xs, np.where(st, us, np.nan), "r-", lw=2.5, label="stable equilibria")
a1.plot(xs, np.where(~st, us, np.nan), "k-", lw=1, label="unstable equilibria")
for o in orbits.values():
    ox = np.array([p.x for p in o])
    ost = np.array([p.stable for p in o])
    for f in ("umax", "umin"):
        y = np.array([getattr(p, f)["x"] for p in o])
        a1.plot(ox[ost], y[ost], "g.", ms=3)
        a1.plot(ox[~ost], y[~ost], ".", color="tab:blue", ms=2)
a1.plot([], [], "g.", label="stable orbits (max/min x)")
a1.plot([], [], ".", color="tab:blue", label="unstable orbits")
a1.plot([p.x for p in hbs], [p.u0["x"] for p in hbs], "o", color="purple", ms=7, label="Hopf")
a1.plot([p.x for p in lps], [p.u0["x"] for p in lps], "s", color="tab:orange", ms=7, label="fold")
a1.set(xlim=(-3.5, 3), ylim=(-2.6, 2.6), xlabel="a", ylabel="x", title="Equilibria and orbits, b = 3")
a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
bb = np.linspace(0, 5, 200)
a2.plot((2 * bb / 3) * np.sqrt(bb / 3), bb, color="0.75", lw=6, label="a = +-(2b/3) sqrt(b/3)")
a2.plot(-(2 * bb / 3) * np.sqrt(bb / 3), bb, color="0.75", lw=6)
for c in fold:
    a2.plot([p.x for p in c], [p.par["b"] for p in c], "k-", lw=1.5)
a2.plot([], [], "k-", label="fold curve (AUTO)")
a2.plot(lps[0].x, 3, "s", color="tab:orange", ms=7)
a2.set(xlim=(-3.5, 3.5), ylim=(-0.2, 5), xlabel="a", ylabel="b", title="Fold of equilibria in (a, b)")
a2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
fig.tight_layout()
fig.savefig(FIG / "bifurcation.png", dpi=130)

# %% 5. simulations either side of the Hopf point at a = -1.8224 (upper branch)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
sim = {}
for a_val, col in ((-1.9, "tab:red"), (-1.7, "tab:green")):
    r = np.roots([-1, 0, 3, a_val])                       # equilibria: a + 3x - x^3 = 0
    x_up = r.real[np.argmax(r.real)]
    tr = m.run(keep=False, a=a_val, x=x_up + 0.05, x0=x_up + 0.05, total=200)
    sim[a_val] = (tr, x_up)
    ax.plot(tr.t, tr["x"], color=col, lw=1, label=f"a = {a_val}")
ax.set(xlim=(0, 60), xlabel="t", ylabel="x", title="Near the upper equilibrium")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "simulations.png", dpi=130)

# %% 6. checks
print("\nchecks:")
check("fold at a = +2 (x = -1)", max(p.x for p in lps), 2, 1e-6)
check("fold at a = -2 (x = +1)", min(p.x for p in lps), -2, 1e-6)
on = fb >= 0.05
err = np.max(np.abs(np.abs(fa[on]) - (2 * fb[on] / 3) * np.sqrt(fb[on] / 3)))
check(f"fold curve: largest | |a| - (2b/3) sqrt(b/3) | on {on.sum()} points", err, 0, 1e-6)
check("fold curve: largest | x^2 - b/3 |", np.max(np.abs(fx[on]**2 - fb[on] / 3)), 0, 1e-5)
# Hopf: c = b - 3x^2 = -k pi/2 (k = 1, 5) gives roots +-i k pi/2 of lambda = c exp(-lambda);
# then a = x^3 - 3x
for k, per in ((1, 4.0), (5, 0.8)):
    x_h = np.sqrt((3 + k * np.pi / 2) / 3)
    a_h = x_h**3 - 3 * x_h
    for sgn in (1, -1):
        h = min(hbs, key=lambda p: abs(p.x - sgn * a_h))
        check(f"Hopf at a = {sgn * a_h:+.6f} (x = {sgn * x_h:+.6f})", h.x, sgn * a_h, 2e-5)
        check("  period there", h.period, per, 1e-3 * per)
up = sorted(hbs, key=lambda p: p.x)[0]               # the Hopf point with x = +1.234, a = -1.822
o4 = orbits[up.label]
check("period-4 orbits are stable (1 = yes)", float(all(p.stable for p in o4)), 1, 0)
check("their period grows past 20 (end on a saddle loop)", float(o4[-1].period > 20), 1, 0)
tr, xu = sim[-1.9]
check("a = -1.9: simulation settles on the equilibrium", tr["x"][-1], xu, 1e-4)
tr, xu = sim[-1.7]
x = tr["x"][tr.t > 100]
t = tr.t[tr.t > 100]
upc = np.where((x[:-1] < xu) & (x[1:] >= xu))[0]
tc = t[upc] + (xu - x[upc]) / (x[upc + 1] - x[upc]) * (t[upc + 1] - t[upc])
T_sim = np.diff(tc).mean()
k4 = int(np.argmax([p.x for p in o4])) + 1         # up to the turning point near the saddle loop
T_auto = np.interp(-1.7, [p.x for p in o4[:k4]], [p.period for p in o4[:k4]])
check("a = -1.7: period of the simulated oscillation vs AUTO", T_sim, T_auto, 1e-2)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
