"""
Lorenz 84: a model of the atmosphere
====================================

Lorenz's 1984 low-order model of the general circulation: x is the strength of
the westerly wind (the jet), y and z the cosine and sine phases of a chain of
travelling eddies.  F is the thermal forcing of the jet and G the forcing of the
eddies.  With a = 0.25, b = 4, G = 1 the flow is chaotic at F = 8 and periodic
at F = 6; at small F the steady westerly flow is stable.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. compares the chaotic flow at F = 8 with the periodic flow at F = 6
  3. follows the steady flow in F with AUTO (G = 1): two folds and a Hopf point
  4. sets G = 0 and follows the Hadley state x = F, y = z = 0, which loses
     stability in a Hopf bifurcation at F = 1
  5. checks the results against exact values for this model

Run:   python 3D-FLOWS_lorenz-84.py
Figures go to figures/3D-FLOWS_lorenz-84/.

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


def rhs(u, p):
    """The right-hand side again, in numpy, to test AUTO's equilibria."""
    x, y, z = u["x"], u["y"], u["z"]
    return np.array([-y**2 - z**2 - p["a"] * x + p["a"] * p["f"],
                     x * y - p["b"] * x * z - y + p["g"],
                     p["b"] * x * y + x * z - z])


def jac(u, p):
    x, y, z, a, b = u["x"], u["y"], u["z"], p["a"], p["b"]
    return np.array([[-a, -2 * y, -2 * z],
                     [y - b * z, x - 1, -b * x],
                     [b * y + z, b * x, x - 1]])


# %% 1. the model
mb = (ModelBuilder("lorenz84")
      .comment("Lorenz 84 general circulation model")
      .ode("x", "-y^2-z^2-a*x+a*f")
      .ode("y", "x*y-b*x*z-y+g")
      .ode("z", "b*x*y+x*z-z")
      .parameter(a=0.25, b=4, f=8, g=1)
      .initial(x=1, y=0, z=0)
      .option(total=200, dt=0.02, trans=20))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. chaotic (F = 8) and periodic (F = 6) flow
fig = plt.figure(figsize=(11, 4.8))
spread = {}
for k, (F, col) in enumerate(((8, "tab:blue"), (6, "tab:red"))):
    tr = m.run(keep=False, f=F, total=600, trans=300)
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.plot(tr["y"], tr["z"], tr["x"], lw=0.4, color=col)
    ax.set(xlabel="y", ylabel="z", zlabel="x (westerly wind)",
           title=f"F = {F}: {'chaotic' if F == 8 else 'periodic'}")
    x = tr["x"]
    mx = x[np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1]
    spread[F] = np.ptp(mx)                # one value per cycle repeats if periodic
    print(f"F = {F}: {len(mx)} maxima of x, spread {spread[F]:.4f}")
fig.tight_layout()
fig.savefig(FIG / "flows.png", dpi=130)

# %% 3. AUTO: the steady flow in F at G = 1
m.set(f=0.1, g=1)
m.run(total=500)                          # settle on the stable steady flow at F = 0.1
m.run_last(total=10)
print("AUTO's default tolerances:", {k: m.auto.settings[k] for k in ("epsl", "epsu", "epss")})
# tighter tolerances than the default, so that the equilibria (and the special
# points, located by AUTO's own iteration) satisfy the equations to ~1e-8
eqs = m.auto.run("f", start="steady", ds=0.01, dsmax=0.05, parmin=0, parmax=10, nmax=1000,
                 epsl=1e-8, epsu=1e-8, epss=1e-6)
lps, hbs = eqs.special("LP"), eqs.special("HB")
print("\nAUTO, steady flow in F (G = 1):")
print("  folds at F =", [round(p.x, 5) for p in lps], "; Hopf at F =", [round(p.x, 5) for p in hbs])
for p in hbs:
    l1, om = m.auto.lyapunov(p)
    print(f"  Hopf at F = {p.x:.5f}: l1 = {l1:+.4f} ({'sub' if l1 > 0 else 'super'}critical), "
          f"period at onset {2 * np.pi / om:.3f}")
resid = max(np.abs(rhs(p.u0, p.par)).max() for p in eqs)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
fs = np.array([p.x for p in eqs])
xs = np.array([p.u0["x"] for p in eqs])
st = np.array([p.stable for p in eqs])
ax.plot(fs, np.where(st, xs, np.nan), "r-", lw=2.5, label="stable steady flow")
ax.plot(fs, np.where(~st, xs, np.nan), "k-", lw=1, label="unstable steady flow")
ax.plot([p.x for p in lps], [p.u0["x"] for p in lps], "s", color="tab:green", ms=7, label="fold (LP)")
ax.plot([p.x for p in hbs], [p.u0["x"] for p in hbs], "o", color="purple", ms=8, label="Hopf")
ax.set(xlim=(0, 10), xlabel="F", ylabel="x (westerly wind)", title="Steady flow of Lorenz 84, G = 1")
ax.legend(fontsize=8, loc="upper left")
zoom = ax.inset_axes([0.55, 0.08, 0.42, 0.42])          # the fold and the Hopf point close up
zoom.plot(fs, np.where(st, xs, np.nan), "r-", lw=2.5)
zoom.plot(fs, np.where(~st, xs, np.nan), "k-", lw=1)
zoom.plot([p.x for p in lps], [p.u0["x"] for p in lps], "s", color="tab:green", ms=6)
zoom.plot([p.x for p in hbs], [p.u0["x"] for p in hbs], "o", color="purple", ms=6)
zoom.set(xlim=(1.15, 1.4), ylim=(0.5, 1.25))
zoom.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "bifurcation.png", dpi=130)

# %% 4. G = 0: the Hadley circulation x = F, y = z = 0
m.set(g=0, f=0.5, x=0.5, y=0, z=0)        # start exactly on it
m.run(total=50)
m.run_last(total=10)
m.auto.reset()                            # start a new diagram (see the note at the end)
had = m.auto.run("f", start="steady", ds=0.01, dsmax=0.05, parmin=0, parmax=3, nmax=500)
hb0 = had.special("HB")[0]
l1, om0 = m.auto.lyapunov(hb0)
print(f"\nG = 0: Hadley state loses stability at F = {hb0.x:.6f}, frequency {om0:.5f}")
hadley_err = max(max(abs(p.u0["x"] - p.x), abs(p.u0["y"]), abs(p.u0["z"])) for p in had)

# %% 5. checks
print("\nchecks:")
check("largest |right-hand side| on AUTO's steady flow (exact 0)", resid, 0, 1e-6)
# exact: at a fold one eigenvalue is 0; at a Hopf point a pair is +-i w
for p in lps:
    check(f"smallest |eigenvalue| at the fold F = {p.x:.4f} (exact 0)",
          np.abs(np.linalg.eigvals(jac(p.u0, p.par))).min(), 0, 1e-5)
ev = np.linalg.eigvals(jac(hbs[0].u0, hbs[0].par))
pair = ev[np.argmax(np.abs(ev.imag))]
check("real part of the complex pair at the Hopf point (exact 0)", pair.real, 0, 1e-5)
check("spread of maxima of x at F = 6 (periodic: 0)", spread[6], 0, 1e-3)
check("spread of maxima of x at F = 8 is large (chaos: 1 = yes)", float(spread[8] > 0.5), 1, 0)
# exact (G = 0): y = z = 0 gives x = F.  The eddies there obey
#   y' = (x-1) y - b x z,  z' = b x y + (x-1) z,
# eigenvalues (x-1) +- i b x, so they grow once x = F > 1, with frequency b F = 4 at onset
check("Hadley state is x = F, y = z = 0 along the branch", hadley_err, 0, 1e-8)
check("Hopf of the Hadley state at F = 1 (exact)", hb0.x, 1.0, 1e-6)
check("frequency at onset b*F = 4 (exact)", om0, 4.0, 1e-4)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")
# Note: m.auto.reset() before the second diagram.  Without it pyxpp drops the new
# points, because XPP numbers them from zero again after discarding the old
# diagram, and pyxpp keeps only points numbered after the old ones.

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
