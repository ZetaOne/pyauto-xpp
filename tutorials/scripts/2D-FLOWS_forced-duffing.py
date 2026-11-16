"""
Forced Duffing oscillator: period doubling to chaos
===================================================

A ball in a double well (a = -1, b = 1) with damping d, shaken periodically:
x' = y, y' = -d y - a x - b x^3 + f cos(om t).  The right-hand side depends on
time, so a phase-plane picture with nullclines and a direction field means
nothing here.  Instead we look once per forcing period T = 2*pi/om (a
stroboscopic, or Poincare, section): a periodic response becomes a fixed point
of this map, a response of period 2T two points, and chaos a fractal cloud.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. draws the stroboscopic section of the chaotic attractor at f = 0.3
  3. computes the stroboscopic bifurcation diagram in f (following the attractor
     as f decreases), with a zoom on the period-doubling cascade
  4. finds the fixed point of the stroboscopic map by Newton's method, its
     Floquet multipliers (from the variational equations), and the f at which one passes through -1 (the first
     period doubling)
  5. checks the map against exact results (linear response, Liouville)

Run:   python 2D-FLOWS_forced-duffing.py
Figures go to figures/2D-FLOWS_forced-duffing/.

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
    NAME = "2D-FLOWS_forced-duffing"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
mb = (ModelBuilder("forced_duffing")
      .comment("periodically forced double-well Duffing oscillator")
      .ode("x", "y")
      .ode("y", "-d*y-a*x-b*x^3+f*cos(om*t)")
      .parameter(d=0.15, a=-1, b=1, f=0.3, om=1.2)
      .initial(x=1, y=0)
      .option(total=200, dt=0.02, meth="rk4"))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
d, om = 0.15, 1.2
T = 2 * np.pi / om
STROBE = dict(dt=T / 200, nout=200)       # 200 rk4 steps per period, store one point per period


def strobe(f, x, y, periods, skip=0):
    """the attractor seen once per period: (x, y) at t = skip*T, ..., (skip+periods)*T"""
    tr = m.run(keep=False, f=f, x=x, y=y, total=(skip + periods) * T, trans=skip * T, **STROBE)
    return tr["x"], tr["y"]


# %% 2. stroboscopic section of the chaotic attractor at f = 0.3
sx, sy = strobe(0.3, 1, 0, 4000, skip=50)
flow = m.run(keep=False, f=0.3, total=60 * T, trans=50 * T, dt=T / 200)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(flow["x"], flow["y"], color="0.8", lw=0.6, label="the flow (10 periods)")
ax.plot(sx, sy, ".", color="tab:blue", ms=1.5, label="once per period (4000 points)")
ax.set(xlim=(-1.8, 1.8), ylim=(-1.2, 1.2), xlabel="x", ylabel="y",
       title="Forced Duffing, f = 0.3: stroboscopic section")
ax.legend(fontsize=8, loc="upper left", markerscale=6)
fig.tight_layout()
fig.savefig(FIG / "strobe_section.png", dpi=130)


# %% 3. stroboscopic bifurcation diagram in f, following the attractor as f decreases
def diagram(F, x, y, skip, keep):
    pts = []
    for f in F:
        sx, sy = strobe(f, x, y, keep, skip)
        x, y = sx[-1], sy[-1]                  # next f starts on this attractor (same forcing phase)
        pts.append(sx[1:])
    return pts


F_all = np.linspace(0.45, 0.1, 351)
F_zoom = np.linspace(0.225, 0.185, 201)
D_all = diagram(F_all, 1.0, 0.0, 120, 80)
D_zoom = diagram(F_zoom, 0.5, 0.0, 400, 64)


# %% 4. the stroboscopic map P, its fixed point and Floquet multipliers
# XPP stores trajectories in single precision, too coarse for finite differences of P.  So a
# second model carries the variational equations: the columns (u1, v1), (u2, v2) of the
# Jacobian DP obey u' = v, v' = -d v - (a + 3 b x^2) u, starting from the identity.
var = (ModelBuilder("forced_duffing_variational")
       .ode("x", "y").ode("y", "-d*y-a*x-b*x^3+f*cos(om*t)")
       .ode("u1", "v1").ode("v1", "-d*v1-(a+3*b*x^2)*u1")
       .ode("u2", "v2").ode("v2", "-d*v2-(a+3*b*x^2)*u2")
       .parameter(d=d, a=-1, b=1, f=0.3, om=om)
       .initial(x=1, y=0, u1=1, v1=0, u2=0, v2=1)
       .option(total=T, dt=T / 400, meth="rk4"))
mv = var.load(HERE / "odes" / f"{NAME}_variational.ode")


def P_and_DP(f, z):
    tr = mv.run(keep=False, f=f, x=z[0], y=z[1], total=T, dt=T / 400)
    e = {k: tr[k][-1] for k in ("x", "y", "u1", "v1", "u2", "v2")}
    return np.array([e["x"], e["y"]]), np.array([[e["u1"], e["u2"]], [e["v1"], e["v2"]]])


def multipliers(f, z):
    for _ in range(8):                        # Newton on P(z) - z = 0
        Pz, J = P_and_DP(f, z)
        z = z + np.linalg.solve(J - np.eye(2), z - Pz)
    return z, np.linalg.eigvals(P_and_DP(f, z)[1])


def fixed_point(f, z):
    return multipliers(f, z)[0]


z = np.array(strobe(0.18, 0.4, 0, 1, skip=300))[:, -1]
f_lo, f_hi = 0.18, 0.215                  # smallest multiplier > -1 at f_lo, < -1 at f_hi
for _ in range(30):
    fm = (f_lo + f_hi) / 2
    zm, mu = multipliers(fm, z)
    if np.real(mu).min() > -1:
        f_lo, z = fm, zm
    else:
        f_hi = fm
f_pd = (f_lo + f_hi) / 2
z_pd, mu_pd = multipliers(f_pd, z)
print(f"\nfirst period doubling at f = {f_pd:.6f}: fixed point {np.round(z_pd, 5)}, "
      f"multipliers {np.round(np.real(mu_pd), 6)}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
for ax, F, D in ((a1, F_all, D_all), (a2, F_zoom, D_zoom)):
    for f, xs in zip(F, D):
        ax.plot(np.full(len(xs), f), xs, ".", color="k", ms=0.8)
    ax.set(xlabel="forcing amplitude f", ylabel="x at t = nT", xlim=(F.min(), F.max()))
a1.axvspan(F_zoom.min(), F_zoom.max(), color="tab:orange", alpha=0.15, label="zoom on the right")
a1.legend(fontsize=8, loc="lower left")
a1.set_title("Stroboscopic bifurcation diagram (f decreasing)")
a2.axvline(f_pd, color="tab:red", lw=1, ls="--", label=f"multiplier = -1 at f = {f_pd:.4f}")
a2.set(ylim=(0, 0.75), title="Period doubling to chaos")
a2.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "bifurcation_diagram.png", dpi=130)

# %% 5. checks
print("\nchecks:")
# Liouville: div of the vector field is -d, so P shrinks areas by exactly exp(-d T): det DP = exp(-dT)
check("det of the strobe map's Jacobian = exp(-d T) (exact)",
      np.linalg.det(P_and_DP(0.3, np.array([0.5, 0.2]))[1]), np.exp(-d * T), 1e-6)
check("product of Floquet multipliers at the doubling = exp(-d T)", np.real(np.prod(mu_pd)), np.exp(-d * T), 1e-6)
check("other multiplier at the doubling = -exp(-d T)", np.real(mu_pd).max(), -np.exp(-d * T), 1e-4)
# weak forcing near the well bottom x = 1: xi = x - 1 obeys xi'' + d xi' + 2 xi = f cos(om t)
# (the linearised spring is -(a + 3b) = 2), so xi = A cos(om t - phi), A = f / sqrt((2-om^2)^2 + (d om)^2),
# tan phi = d om / (2 - om^2); at t = nT: xi = A cos phi, xi' = A om sin phi.  Neglected: O(A^2) ~ 3e-6
fs = 1e-3
A, phi = fs / np.hypot(2 - om**2, d * om), np.arctan2(d * om, 2 - om**2)
z_lin = fixed_point(fs, np.array([1.0, 0.0]))
check("weak forcing: strobe x - 1 vs linear theory", z_lin[0] - 1, A * np.cos(phi), 1e-5)
check("weak forcing: strobe y vs linear theory", z_lin[1], A * om * np.sin(phi), 1e-5)
# brute force: just above a supercritical doubling the two strobe points split by s ~ sqrt(f - f_pd),
# so s^2 is linear in f and vanishes at f_pd.  (Counting points fails: next to f_pd the multiplier
# is near -1 and the approach to the fixed point alternates for thousands of periods.)
split = np.array([abs(xs[-1] - xs[-2]) for xs in D_zoom])
near = (split > 0.02) & (F_zoom < f_pd + 0.006)
slope, icpt = np.polyfit(F_zoom[near], split[near] ** 2, 1)
check("diagram: zero of the split^2 fit vs Floquet doubling", -icpt / slope, f_pd, F_zoom[0] - F_zoom[1])
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
