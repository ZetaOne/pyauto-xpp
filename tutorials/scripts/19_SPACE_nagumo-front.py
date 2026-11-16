"""
Nagumo travelling front: shooting for a heteroclinic orbit
==========================================================

The bistable cable equation u_t = u_xx + u(1-u)(u-a) has fronts that turn the
state u = 1 into u = 0 (or back), travelling at a fixed speed c.  In the moving
frame z = x - c t a front u(z) solves u'' + c u' + u(1-u)(u-a) = 0, written
here as  u' = v,  v' = -c v - u(1-u)(u-a).  Both (1, 0) and (0, 0) are saddles,
and a front is an orbit leaving one and landing on the other: a heteroclinic
orbit, which exists only for one value of c.  The exact answer (Huxley's front)
is c = (1 - 2a)/sqrt(2) with profile u = 1/(1 + exp(z/sqrt(2))).

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. finds the speed by shooting: leave the saddle at (1, 0) along its unstable
     eigenvector (from XPP's eigenvalues) and bisect on c between orbits that
     overshoot u = 0 and orbits that turn back
  3. repeats this for several a and compares with the exact speed
  4. draws the heteroclinic orbit in the phase plane with the nullclines, and
     the front profile u(z) against the exact one
  5. checks the results against the exact solution

Run:   python 19_SPACE_nagumo-front.py
Figures go to figures/19_SPACE_nagumo-front/.

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
    NAME = "19_SPACE_nagumo-front"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)
DELTA = 1e-6                              # distance from the saddle where the orbit starts


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
mb = (ModelBuilder("nagumo_front")
      .comment("Nagumo front in the moving frame: u'' + c u' + u(1-u)(u-a) = 0")
      .ode("u", "v")
      .ode("v", "-c*v-u*(1-u)*(u-a)")
      .parameter(c=0.3, a=0.25)
      .initial(u=0.9, v=0)
      .option(total=40, dt=0.05))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
m.set(jac_eps=1e-8)      # Jacobian step for eigenvalues (XPP 8.1 ignores "@ jac_eps" in the file)


# %% 2. shooting from the saddle at (1, 0)
def shoot(c, a, total=200):
    """Leave (1, 0) towards u < 1 along the unstable eigenvector (1, lam)."""
    m.set(c=c, a=a)
    lam = max(m.equilibrium({"u": 1, "v": 0}).eigenvalues.real)
    return m.run(keep=False, u=1 - DELTA, v=-DELTA * lam, total=total), lam


def front_speed(a, lo=0.0, hi=1.0, steps=40):
    """Bisect on c: below the speed the orbit overshoots u = 0, above it turns back."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        tr, _ = shoot(mid, a)
        lo, hi = (mid, hi) if tr["u"].min() < 0 else (lo, mid)
    return 0.5 * (lo + hi)


# show the two kinds of miss for a = 0.25 on either side of the speed
A0 = 0.25
c0 = front_speed(A0)
tr0, lam0 = shoot(c0, A0)
misses = {f"c = {c0 - 0.05:.3f} (too slow)": shoot(c0 - 0.05, A0, total=40)[0],
          f"c = {c0 + 0.05:.3f} (too fast)": shoot(c0 + 0.05, A0, total=40)[0]}
print(f"\na = {A0}: front speed {c0:.10f}; unstable eigenvalue at (1, 0) {lam0:.10f}")

# %% 3. the speed for several a
avals = np.array([0.1, 0.2, 0.25, 0.3, 0.4])
cs = np.array([front_speed(a) for a in avals])
exact = (1 - 2 * avals) / np.sqrt(2)
print("\n   a     shooting       exact (1-2a)/sqrt(2)")
for a, c, e in zip(avals, cs, exact):
    print(f"  {a:4.2f}  {c:.10f}  {e:.10f}")

fig, ax = plt.subplots(figsize=(5.2, 3.8))
aa = np.linspace(0, 0.5, 50)
ax.plot(aa, (1 - 2 * aa) / np.sqrt(2), "k-", lw=1, label="exact (1 - 2a)/sqrt(2)")
ax.plot(avals, cs, "o", color="tab:red", label="shooting with XPP")
ax.set(xlabel="a", ylabel="front speed c", title="Speed of the Nagumo front")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "speed_vs_a.png", dpi=130)

# %% 4. the heteroclinic orbit and the front profile (a = 0.25)
u, v, z = tr0["u"], tr0["v"], tr0.t
on = np.arange(len(u)) <= np.argmax(u < 1e-3)         # up to where the orbit nears (0, 0)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
uu = np.linspace(-0.2, 1.2, 300)
ax.axhline(0, color="tab:orange", lw=2, label="u' = 0")
ax.plot(uu, -uu * (1 - uu) * (uu - A0) / c0, color="tab:green", lw=2, label="v' = 0")
for (lab, tr), col in zip(misses.items(), ("tab:purple", "tab:brown")):
    ax.plot(tr["u"], tr["v"], "--", lw=1, color=col, label=lab)
ax.plot(u[on], v[on], color="tab:blue", lw=2.5, label=f"heteroclinic orbit, c = {c0:.4f}")
ax.plot(uu[(uu >= 0) & (uu <= 1)], -uu[(uu >= 0) & (uu <= 1)] * (1 - uu[(uu >= 0) & (uu <= 1)]) / np.sqrt(2),
        "k:", lw=1.5, label="exact: v = -u(1-u)/sqrt(2)")
ax.plot([0, 1], [0, 0], "ko", ms=7)
ax.set(xlim=(-0.2, 1.2), ylim=(-0.3, 0.1), xlabel="u", ylabel="v = u'", title=f"Nagumo front, a = {A0}")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
fig.tight_layout()
fig.savefig(FIG / "phase_plane.png", dpi=130)

k = np.argmax(u < 0.5)                                 # put u = 1/2 at z = 0
z0 = z[k - 1] + (0.5 - u[k - 1]) / (u[k] - u[k - 1]) * (z[k] - z[k - 1])
zs = z - z0
profile = 1 / (1 + np.exp(zs / np.sqrt(2)))
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(zs[on], u[on], color="tab:blue", lw=2.5, label="XPP, shot from (1, 0)")
ax.plot(zs[on], profile[on], "k:", lw=1.5, label="exact 1/(1 + exp(z/sqrt(2)))")
ax.set(xlim=(-12, 10), xlabel="z = x - c t", ylabel="u", title="Front profile")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "profile.png", dpi=130)

# %% 5. checks
print("\nchecks:")
# the bisection pins c to 1e-12; what is left comes from the start DELTA off the saddle and the
# Runge-Kutta step (dt = 0.05); both are far below 1e-6
for a, c, e in zip(avals, cs, exact):
    check(f"speed at a = {a} vs (1-2a)/sqrt(2) (exact)", c, e, 1e-6)
# exact: at c = (1-2a)/sqrt(2) the eigenvalues at (1, 0) solve lam^2 + c lam - (1-a) = 0,
# whose positive root is 1/sqrt(2) for every a (the front decays like exp(-z/sqrt(2)))
check("unstable eigenvalue at (1, 0) = 1/sqrt(2) (exact)", lam0, 1 / np.sqrt(2), 1e-6)
# along the orbit, v = u' = -u(1-u)/sqrt(2) exactly; the profile is the logistic curve.
# Runge-Kutta with dt = 0.05 has a global error of order dt^4 ~ 6e-6: allow 1e-5
check("largest |v + u(1-u)/sqrt(2)| on the orbit", np.abs(v[on] + u[on] * (1 - u[on]) / np.sqrt(2)).max(), 0, 1e-5)
check("largest |u(z) - exact profile|", np.abs(u[on] - profile[on]).max(), 0, 1e-5)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
