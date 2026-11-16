"""
Forced phase oscillator: devil's staircase and Arnold tongues
=============================================================

x is the phase of an oscillator with natural frequency 1, y the phase of a
periodic forcing with frequency om:  x' = 1 + b p(x) cos y,  y' = om, where
p(x) = -sin x / (1 + a cos x) says how strongly the oscillator responds at each
phase.  Both are angles, so the phase space is a torus.  The rotation number
rho (turns of x per turn of the forcing) locks to a rational p/q over whole
intervals of om; these intervals grow with b into the Arnold tongues.

This script
  1. builds the model with ModelBuilder (XPP folds x and y onto the torus) and
     an unfolded copy, whose x keeps counting turns, for rotation numbers
  2. draws orbits on the torus: one locked, one quasi-periodic
  3. computes rho against om at b = 0.4: the devil's staircase
  4. maps the locked regions on an (om, b) grid: the Arnold tongues, with the
     first-order theory for the 1:1 tongue
  5. checks rho at b = 0 (exactly 1/om), the 1:1 plateau, and the 1:1 tongue
     width against averaging theory

Run:   python 2D-FLOWS_forced-phase-oscillator.py
Figures go to figures/2D-FLOWS_forced-phase-oscillator/.

Model from the examples of XPP Studio (Peter Ashwin and Bard Ermentrout).
Every number is computed by XPPAUT 8.1 and its AUTO through pyxpp.
"""
# %% setup
from fractions import Fraction
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
    NAME = "2D-FLOWS_forced-phase-oscillator"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
TWO_PI = 2 * np.pi
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


def broken(*cols):
    """NaN wherever a folded angle wraps, so lines don't cross the torus"""
    cols = [np.asarray(c, float).copy() for c in cols]
    jump = np.zeros(len(cols[0]) - 1, bool)
    for c in cols:
        jump |= np.abs(np.diff(c)) > np.pi
    for c in cols:
        c[1:][jump] = np.nan
    return cols


# %% 1. the model, and an unfolded copy
def build(name, folded):
    mb = (ModelBuilder(name)
          .comment("phase oscillator forced at frequency om")
          .function("p(x)", "-sin(x)/(1+a*cos(x))")
          .ode("x", "1+b*p(x)*cos(y)")
          .ode("y", "om")
          .parameter(om=2, b=0.25, a=0.25)
          .initial(x=0, y=0))
    if folded:
        mb = mb.option(fold="x", tor_per=6.283185).option(fold="y")
    return mb.option(total=100, dt=0.05, bound=1e9)


mb = build("forced_phase_oscillator", folded=True)
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
mu = build("forced_phase_oscillator_unfolded", folded=False).load(HERE / "odes" / f"{NAME}_unfolded.ode")


def rotation_number(om, b, cycles=360, skip=60, k=60):
    """turns of x per forcing turn, from x sampled once per forcing period after `skip` periods;
    `cycles` is a multiple of 1..6, so a p:q locked orbit (q <= 6) gives exactly p/q"""
    Tf = TWO_PI / om
    x = mu.run(keep=False, om=om, b=b, x=0, y=0, total=(skip + cycles + 2) * Tf, trans=skip * Tf,
               dt=Tf / k, nout=k)["x"]
    return (x[cycles] - x[0]) / (TWO_PI * cycles)


# %% 2. orbits on the torus
fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
for ax, (om, b, what) in zip(axs, [(1.05, 0.25, "locked 1:1"), ((1 + 5**0.5) / 2, 0.05, "quasi-periodic")]):
    tr = m.run(keep=False, om=om, b=b, total=250, trans=100)
    ax.plot(*broken(tr["y"], tr["x"]), lw=0.8, color="tab:blue")
    ax.set(xlim=(0, TWO_PI), ylim=(0, TWO_PI), aspect="equal", xlabel="forcing phase y", ylabel="phase x",
           title=f"om = {om:.3f}, b = {b}: {what}, rho = {rotation_number(om, b):.4f}")
    ax.set_xticks([0, np.pi, TWO_PI], ["0", "π", "2π"])
    ax.set_yticks([0, np.pi, TWO_PI], ["0", "π", "2π"])
fig.tight_layout()
fig.savefig(FIG / "torus_orbits.png", dpi=130)

# %% 3. devil's staircase at b = 0.4
OM = np.linspace(0.5, 3.2, 271)
rho = np.array([rotation_number(o, 0.4) for o in OM])
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(OM, 1 / OM, color="0.6", ls=":", label="b = 0: rho = 1/om")
ax.plot(OM, rho, color="tab:blue", lw=1.8, label="b = 0.4")
for frac in (Fraction(1, 1), Fraction(2, 3), Fraction(1, 2), Fraction(1, 3)):
    on = np.abs(rho - float(frac)) < 1e-5
    if on.any():
        ax.plot(OM[on], rho[on], color="tab:red", lw=3.5, solid_capstyle="butt")
        ax.annotate(f"{frac.numerator}:{frac.denominator}", (OM[on].mean(), float(frac)),
                    textcoords="offset points", xytext=(4, 6), color="tab:red", fontsize=9)
ax.plot([], [], color="tab:red", lw=3.5, label="locked plateaus")
ax.set(xlim=(0.5, 3.2), xlabel="forcing frequency om", ylabel="rotation number rho",
       title="Devil's staircase")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "devils_staircase.png", dpi=130)

# %% 4. Arnold tongues on an (om, b) grid
OMg, Bg = np.linspace(0.5, 3.2, 181), np.linspace(0, 0.5, 21)
R = np.array([[rotation_number(o, bb, cycles=120, skip=200) for o in OMg] for bb in Bg])
tongues = [Fraction(1, 1), Fraction(2, 3), Fraction(1, 2), Fraction(1, 3)]
locked = np.full(R.shape, np.nan)
for n, frac in enumerate(tongues):
    locked[np.abs(R - float(frac)) < 1e-5] = n          # at b = 0 only om = q/p exactly: the tips
# averaging (first order in b): psi = x - y obeys psi' = 1 - om + b H(psi) with
# H(psi) = <p(u) cos(u - psi)> = -c sin(psi),  c = <sin^2 u / (1 + a cos u)> = (1 - sqrt(1 - a^2)) / a^2,
# so 1:1 locking needs |om - 1| <= c b
a = 0.25
c = (1 - np.sqrt(1 - a**2)) / a**2
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.pcolormesh(OMg, Bg, R, cmap="Greys", shading="nearest", vmin=0, vmax=2.2)
cmap = plt.get_cmap("tab10", 10)
ax.pcolormesh(OMg, Bg, locked, cmap=cmap, vmin=-0.5, vmax=9.5, shading="nearest")
for n, frac in enumerate(tongues):
    ax.plot([], [], "s", color=cmap(n), label=f"{frac.numerator}:{frac.denominator} locked")
bl = np.linspace(0, 0.5, 50)
ax.plot(1 - c * bl, bl, "k--", lw=1.2, label="1:1 edges, first order")
ax.plot(1 + c * bl, bl, "k--", lw=1.2)
ax.set(xlim=(0.5, 3.2), ylim=(0, 0.5), xlabel="forcing frequency om", ylabel="forcing strength b",
       title="Arnold tongues (grey: unlocked, shade = rho)")
ax.text(1.4, 0.015, "tongues narrower than the grid step (0.015 in om) show up broken",
        fontsize=7.5, color="0.25")
ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.62, 0.5), framealpha=0.95)
fig.tight_layout()
fig.savefig(FIG / "arnold_tongues.png", dpi=130)


# %% 5. checks
def tongue_edge(b, side):
    """om at the edge of the 1:1 tongue, by bisection from om = 1 (locked) outwards; near the
    edge the approach to the locked orbit is slow, so use a long transient and many cycles"""
    lo, hi = 1.0, 1 + side * 2 * c * b
    for _ in range(22):
        mid = (lo + hi) / 2
        locked = abs(rotation_number(mid, b, cycles=600, skip=300) - 1) < 1e-5
        lo, hi = (mid, hi) if locked else (lo, mid)
    return (lo + hi) / 2


print("\nchecks:")
# b = 0: x = t and y = om t exactly, so rho = 1/om
check("rho at b = 0, om = 1.3 is 1/om (exact)", rotation_number(1.3, 0), 1 / 1.3, 1e-6)
check("rho at b = 0, om = 2.7 is 1/om (exact)", rotation_number(2.7, 0), 1 / 2.7, 1e-6)
plateau = OM[np.abs(rho - 1) < 1e-6]
check("1:1 plateau at b = 0.4 contains om = 1 (rho there)", rotation_number(1.0, 0.4), 1.0, 1e-6)
check("om = 1 lies inside the 1:1 plateau (1 = yes)", float(plateau.min() < 1 < plateau.max()), 1, 0)
# first-order width 2 c b; the edges move by O(b^2), so the relative error of the width is O(b^2)
bb = 0.05
width = tongue_edge(bb, +1) - tongue_edge(bb, -1)
check("1:1 tongue width / (2 c b) at b = 0.05 (averaging)", width / (2 * c * bb), 1.0, bb**2)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
