"""
Ginzburg-Landau oscillator: phase reduction
===========================================

The complex Ginzburg-Landau (lambda-omega) oscillator in real coordinates, as
in Wilson & Ermentrout, Phys. Rev. Lett. 123, 164101 (2019):
    x' = x (1 - r^2) - q r^2 y,   y' = y (1 - r^2) + q r^2 x,   r^2 = x^2 + y^2.
In polar form r' = r (1 - r^2), theta' = q r^2: the limit cycle is the unit
circle, traversed as (cos qt, sin qt) with period 2*pi/q.  Its phase response
(adjoint), its isostable rate and the interaction function H for the paper's
coupling are all known exactly, which makes it a good test of XPP 8.1's phase
reduction tools (nUmerics > Averaging).

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. finds one period with Initialconds > Periodic, for q = 1 and q = 1.5
  3. computes the adjoint with XPP and compares it with the exact adjoint
  4. computes H for the coupling (x'-x) - d(y'-y), (y'-y) + d(x'-x) and compares
     it with the exact H
  5. runs XPP 8.1's second-order reduction (H2 and gamma) and reports it
  6. checks period, adjoint, H and the isostable rate against exact values

Run:   python 2D-FLOWS_coupled-ginzburg-landau.py
Figures go to figures/2D-FLOWS_coupled-ginzburg-landau/.

Model from the examples of XPP Studio (Peter Ashwin and Bard Ermentrout).
Every number is computed by XPPAUT 8.1 and its AUTO through pyxpp.
"""
# %% setup
import re
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
    NAME = "2D-FLOWS_coupled-ginzburg-landau"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
mb = (ModelBuilder("ginzburg_landau")
      .comment("complex Ginzburg-Landau oscillator in real coordinates")
      .ode("x", "x*(1-x^2-y^2)-q*(x^2+y^2)*y")
      .ode("y", "y*(1-x^2-y^2)+q*(x^2+y^2)*x")
      .parameter(q=1, d=0.9)
      .initial(x=1, y=0)
      .option(total=40, dt=0.01, meth="rk4"))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
d = 0.9
A = m.buttons.numerics.averaging                     # XPP 8.1: nUmerics > Averaging


# Exact results.  The phase phi = theta + q ln r advances at the constant rate q (phi' = q r^2
# + q r'/r = q), so its level sets are the isochrons.  On r = 1, grad(theta) = (-sin, cos) and
# grad(r) = (cos, sin); XPP normalises the adjoint so that Z . F = 1 (phase measured in time),
# i.e. Z = grad(phi)/q:   Z(t) = (q cos qt - sin qt, cos qt + q sin qt) / q.
# In complex form Z = (q + i) e^{i qt}/q and the coupling is G = (1 + i d)(z' - z), z' = e^{iq(t+s)},
# so Z.G = Re[conj(Z) G] = Re[(q - i)(1 + i d)(e^{iqs} - 1)]/q does not depend on t, and
#   H(s) = [(q + d)(cos qs - 1) - (q d - 1) sin qs] / q.
def Z_exact(t, q):
    return (q * np.cos(q * t) - np.sin(q * t)) / q, (np.cos(q * t) + q * np.sin(q * t)) / q


def H_exact(s, q):
    return ((q + d) * (np.cos(q * s) - 1) - (q * d - 1) * np.sin(q * s)) / q


# %% 2.-4. one period, the adjoint and H, for two values of q
res = {}
for q in (1.0, 1.5):
    m.set(q=q, x=1, y=0, total=50, trans=0)
    m.run()
    m.buttons.initialconds.periodic()        # orbit starts at the maximum of x: theta = 0
    T = m.numerics["total"]
    n = len(m.log)
    A.new_adj()
    adj = m.data                             # columns: t, adjoint x, adjoint y
    log_adj = m.log[n:]
    A.make_h("(x'-x)-d*(y'-y)", "(y'-y)+d*(x'-x)")
    H = m.data                               # first column after t is H
    A.orbit()                                # put the orbit back in the data browser
    res[q] = dict(T=T, adj=adj, H=H, log=log_adj)
    print(f"\nq = {q}: period {T:.10f} (exact 2*pi/q = {2 * np.pi / q:.10f})")
    print("  XPP:", log_adj.strip().splitlines()[0].strip())

fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, q in zip(axs[:2], (1.0, 1.5)):
    adj = res[q]["adj"]
    zx, zy = Z_exact(adj.t, q)
    ax.plot(adj.t, zx, color="tab:blue", lw=4, alpha=0.35, label="exact Z_x")
    ax.plot(adj.t, zy, color="tab:orange", lw=4, alpha=0.35, label="exact Z_y")
    ax.plot(adj.t, adj["x"], color="tab:blue", lw=1.2, ls="--", label="XPP Z_x")
    ax.plot(adj.t, adj["y"], color="tab:orange", lw=1.2, ls="--", label="XPP Z_y")
    ax.set(xlim=(0, res[q]["T"]), ylim=(-1.6, 2.3), xlabel="time along the orbit", title=f"Adjoint, q = {q}")
    ax.legend(fontsize=7, ncol=2, loc="upper center")
H = res[1.0]["H"]
axs[2].plot(H.t, H_exact(H.t, 1.0), color="tab:green", lw=4, alpha=0.35, label="exact H")
axs[2].plot(H.t, H["x"], color="tab:green", lw=1.2, ls="--", label="XPP H")
axs[2].axhline(0, color="0.7", lw=0.8)
axs[2].set(xlim=(0, res[1.0]["T"]), xlabel="phase difference", title=f"H for the paper's coupling, q = 1, d = {d}")
axs[2].legend(fontsize=8, loc="center")
fig.tight_layout()
fig.savefig(FIG / "adjoint_and_H.png", dpi=130)

# %% 5. second order: H2 and gamma (XPP 8.1), at q = 1
m.set(q=1.0, x=1, y=0, total=50, trans=0)
m.run()
m.buttons.initialconds.periodic()
A.new_adj()
A.make_h("(x'-x)-d*(y'-y)", "(y'-y)+d*(x'-x)")
A.orbit()
n = len(m.log)
A.h2()
log_h2 = m.log[n:]
print("\nXPP 8.1, second-order phase reduction (q = 1, d = 0.9):")
print("\n".join("  " + line.strip() for line in log_h2.strip().splitlines()))
gamma = float(re.search(r"gamma = .*?= *([-+0-9.eE]+)", log_h2).group(1))
print(f"  gamma = {gamma}")

# %% 6. checks
print("\nchecks:")
for q in (1.0, 1.5):
    check(f"period at q = {q} vs 2*pi/q (exact)", res[q]["T"], 2 * np.pi / q, 1e-8)
for q in (1.0, 1.5):
    adj = res[q]["adj"]
    zx, zy = Z_exact(adj.t, q)
    err = max(np.abs(adj["x"] - zx).max(), np.abs(adj["y"] - zy).max())
    check(f"adjoint at q = {q}: max |XPP - exact|", err, 0, 1e-5)
for q in (1.0, 1.5):
    H = res[q]["H"]
    check(f"H at q = {q}: max |XPP - exact|", np.abs(H["x"] - H_exact(H.t, q)).max(), 0, 1e-5)
# mean of H over a period: -(q + d)/q
check("mean of H at q = 1 is -(q + d)/q (exact)", res[1.0]["H"]["x"][:-1].mean(), -(1 + d), 1e-5)
# isostable (Floquet) exponent: r' = r(1 - r^2) gives dr' = (1 - 3r^2) dr = -2 dr on r = 1, for any q
kappa = float(re.search(r"kappa = ([-+0-9.eE]+)", res[1.5]["log"]).group(1))
check("isostable rate kappa at q = 1.5 (exact -2)", kappa, -2.0, 1e-6)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
