"""
The Lorenz equations: a strange attractor
=========================================

Lorenz's three-variable model of convection: x is the speed of the rolls, y and
z temperature differences.  For r below 1 the fluid is at rest; at r = 1 two
steady rolls appear in a pitchfork; they lose stability in a Hopf bifurcation
near r = 24.74, and at r = 27 the orbit switches irregularly between the two
wings of the butterfly.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. draws the attractor at r = 27, in 3D and in the (x, z) plane
  3. follows the equilibria in r with AUTO: the pitchfork at r = 1, the two
     branches of steady rolls and their Hopf points
  4. shows sensitive dependence: two orbits 1e-8 apart separate like
     exp(lambda t); lambda is read off the slope
  5. draws Lorenz's return map of successive maxima of z
  6. checks the results against exact values for this model

Run:   python 3D-FLOWS_lorenz.py
Figures go to figures/3D-FLOWS_lorenz/.

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
s, b = 10, 2.666667                       # b = 8/3 as XPP Studio writes it
mb = (ModelBuilder("lorenz")
      .comment("Lorenz equations")
      .ode("x", "s*(-x+y)")
      .ode("y", "r*x-y-x*z")
      .ode("z", "-b*z+x*y")
      .parameter(r=27, s=s, b=b)
      .initial(x=-7.5, y=-3.6, z=30)
      .option(total=40, dt=0.01, meth="rk4", bound=1000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. the attractor at r = 27
tr = m.run(keep=False, total=100, trans=10)
fig = plt.figure(figsize=(11, 4.8))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(tr["x"], tr["y"], tr["z"], lw=0.4, color="tab:blue")
ax.set(xlabel="x", ylabel="y", zlabel="z", title="Lorenz attractor, r = 27")
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(tr["x"], tr["z"], lw=0.4, color="tab:blue")
c = np.sqrt(b * (27 - 1))                                # the two steady rolls (exact)
ax2.plot([-c, c], [26, 26], "ko", ms=6, label="unstable equilibria")
ax2.set(xlabel="x", ylabel="z", title="projection on (x, z)")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "attractor.png", dpi=130)

# %% 3. AUTO: equilibria in r
m.set(r=0.5)
m.run(total=200, x=0.1, y=0.1, z=0.1)    # for r < 1 everything decays to the origin
m.run_last(total=10)
eqs = m.auto.run("r", start="steady", ds=0.02, dsmax=0.2, parmin=0, parmax=40, nmax=500)
# XPP's AUTO switches branches at the pitchfork by itself: the diagram holds the
# origin and both families of steady rolls (x > 0 and x < 0)
bp = eqs.special("BP")[0]
hbs = eqs.special("HB")
m.set(r=27)                              # AUTO leaves r at the end of the branch
print(f"\nAUTO: pitchfork at r = {bp.x:.6f}; Hopf on the rolls at r =",
      sorted({round(p.x, 5) for p in hbs}))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
xs = np.array([p.x for p in eqs])
us = np.array([p.u0["x"] for p in eqs])
st = np.array([p.stable for p in eqs])
ax.plot(xs[st], us[st], "r.", ms=3)
ax.plot(xs[~st], us[~st], "k.", ms=2)
ax.plot([], [], "r.", label="stable equilibria")
ax.plot([], [], "k.", label="unstable equilibria")
ax.plot(bp.x, bp.u0["x"], "s", color="tab:green", ms=8, label="pitchfork (BP)")
ax.plot([p.x for p in hbs], [p.u0["x"] for p in hbs], "o", color="purple", ms=8, label="Hopf")
ax.set(xlim=(0, 40), xlabel="r", ylabel="x", title="Equilibria of the Lorenz equations")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "bifurcation.png", dpi=130)

# %% 4. sensitive dependence: separation of nearby orbits
# 40 pairs started 1e-8 apart at points spread over the attractor.  One pair's
# separation wobbles a lot, so average ln(distance) over the pairs; the average
# grows like lambda t until the distance is as big as the attractor.  XPP keeps
# its data in single precision (about 7 digits), so distances below about 1e-5
# are rounding noise in the stored orbits: fit only between 1e-5 and 1e-1.
long = m.run(keep=False, total=1000, trans=10)
starts = long.array[::1500, 1:4][:40]     # columns x, y, z of points on the attractor
lnd = []
fig, ax = plt.subplots(figsize=(6.5, 4))
for x0, y0, z0 in starts:
    a1 = m.run(keep=False, total=30, x=x0, y=y0, z=z0)
    a2 = m.run(keep=False, total=30, x=x0 + 1e-8, y=y0, z=z0)
    sep = np.sqrt(sum((a1[v] - a2[v])**2 for v in "xyz"))
    ax.semilogy(a1.t, np.where(sep > 1e-5, sep, np.nan), lw=0.5, color="tab:gray", alpha=0.6)
    lnd.append(np.log(np.maximum(sep, 1e-12)))      # stored orbits can be equal at first
t, lnd = a1.t, np.mean(lnd, axis=0)
fit = (lnd > np.log(1e-5)) & (np.cumsum(lnd > np.log(1e-1)) == 0)   # before saturation
lam, c0 = np.polyfit(t[fit], lnd[fit], 1)
ax.axhspan(1e-10, 1e-5, color="tab:gray", alpha=0.15, label="below single-precision noise")
ax.semilogy(t, np.where(lnd > np.log(1e-5), np.exp(lnd), np.nan), color="tab:blue", lw=2,
            label="geometric mean of 40 pairs")
ax.semilogy(t[fit], np.exp(c0 + lam * t[fit]), "k--", lw=1.5, label=f"fit: exp({lam:.3f} t)")
ax.plot(0, 1e-8, "ko", ms=5, label="start: 1e-8 apart")
ax.set(xlim=(-0.5, 30), ylim=(1e-10, 1e2), xlabel="t", ylabel="distance between the two orbits",
       title="Sensitive dependence, r = 27")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "separation.png", dpi=130)
print(f"\nlargest Lyapunov exponent from the slope: lambda = {lam:.3f}")

# %% 5. Lorenz's return map: successive maxima of z
z = m.run(keep=False, total=500, trans=20)["z"]
k = np.where((z[1:-1] > z[:-2]) & (z[1:-1] > z[2:]))[0] + 1
zmax = z[k]
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(zmax[:-1], zmax[1:], "k.", ms=3)
lims = (zmax.min() - 1, zmax.max() + 1)
ax.plot(lims, lims, color="tab:gray", lw=0.8, label="$z_{n+1} = z_n$")
ax.set(xlim=lims, ylim=lims, xlabel="$z_n$ (n-th maximum of z)", ylabel="$z_{n+1}$",
       title="Lorenz return map, r = 27")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "return_map.png", dpi=130)
print(f"{len(zmax)} maxima of z, between {zmax.min():.2f} and {zmax.max():.2f}")

# %% 6. checks
print("\nchecks:")
# exact: the origin has eigenvalue (-(s+1) + sqrt((s-1)^2 + 4 s r))/2, which is zero at r = 1
check("pitchfork at r = 1 (exact)", bp.x, 1.0, 1e-4)
# exact: on the rolls x = y = +-sqrt(b(r-1)), z = r-1, the characteristic polynomial
# l^3 + (s+b+1) l^2 + b(s+r) l + 2 b s (r-1) has a pair +-i w when r = s(s+b+3)/(s-b-1)
r_hopf = s * (s + b + 3) / (s - b - 1)
check("Hopf of the rolls (exact)", hbs[0].x, r_hopf, 1e-3)
eq = m.equilibrium({"x": 8, "y": 8, "z": 26})
check("steady roll x = sqrt(b(r-1)) at r = 27 (exact)", eq.x["X"], np.sqrt(b * 26), 1e-6)
check("steady roll is unstable at r = 27 (0 = unstable)", float(eq.stable), 0, 0)
# literature: the largest Lyapunov exponent of the Lorenz attractor is about 0.9
# (0.906 at r = 28; slightly less at r = 27).  A finite-time estimate from 40 pairs.
check("Lyapunov exponent from the separation (about 0.9)", lam, 0.9, 0.1)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
