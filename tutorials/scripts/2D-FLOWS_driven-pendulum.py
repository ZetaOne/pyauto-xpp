"""
Damped driven pendulum: resting or rotating
===========================================

A pendulum with damping b driven by a constant torque i: th' = v,
v' = i - b v - sin th.  The angle th lives on a circle, so the phase space is a
cylinder and XPP folds th into [0, 2*pi).  Rest states need sin th = i, so they
exist only for i < 1; a rotating orbit exists for i above a smaller value, and in
between the pendulum may rest or rotate depending on its history.  The same
equation describes a Josephson junction and a phase-locked loop.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. finds the two rest states at i = 0.5 and compares them with sin th = i
  3. draws the cylinder at i = 0.5: trajectories coloured by where they end up
  4. sweeps i up and down, each run starting where the last one ended, and
     plots the mean velocity: a hysteresis loop between resting and rotating
  5. checks rest states, the jump points and the mean velocity against theory

Run:   python 2D-FLOWS_driven-pendulum.py
Figures go to figures/2D-FLOWS_driven-pendulum/.

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
TWO_PI = 2 * np.pi
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


def broken(th):
    """th with NaN where XPP's fold wraps it, so lines don't cross the plot"""
    th = np.asarray(th, float).copy()
    th[1:][np.abs(np.diff(th)) > np.pi] = np.nan
    return th


def wraps(tr):
    """indices where th passes 2*pi going forward (the pendulum goes over the top)"""
    return np.where(np.diff(tr["th"]) < -np.pi)[0]


def mean_velocity(tr):
    """mean v over whole turns if the pendulum rotates, else 0 (it rests)"""
    w = wraps(tr)
    return tr["v"][w[0] + 1:w[-1] + 1].mean() if len(w) >= 2 else 0.0


# %% 1. the model
mb = (ModelBuilder("driven_pendulum")
      .comment("damped driven pendulum; th is an angle")
      .ode("th", "v")
      .ode("v", "i-b*v-sin(th)")
      .parameter(i=0.5, b=0.3)
      .initial(th=0, v=2.5)
      .option(fold="th", tor_per=6.283185307, total=100, dt=0.05))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
i0, b = 0.5, 0.3

# %% 2. rest states at i = 0.5
# XPP's Jacobian is a forward difference with relative step jac_eps (default 1e-3, the same
# number as the Newton tolerance), which puts errors of ~1e-4 into the eigenvalues; use 1e-7.
m.set_option("jac_eps", 1e-7)
node = m.equilibrium({"th": 0.5, "v": 0})
saddle = m.equilibrium({"th": 2.6, "v": 0})
print("\nrest states at i = 0.5:")
for e in (node, saddle):
    print(" ", e, " eigenvalues", np.round(e.eigenvalues, 5))
m.set(i=1.2)
try:
    m.equilibrium({"th": 1.5, "v": 0})
    print("  at i = 1.2 Newton found a rest state (unexpected)")
except Exception as err:                         # pyxpp raises XPPError: there is none
    print("  at i = 1.2:", err)
m.set(i=i0)

# %% 3. the cylinder at i = 0.5: rest or rotate?
fig, ax = plt.subplots(figsize=(7, 4.6))
for th0 in np.linspace(0, TWO_PI, 9)[:-1]:
    for v0 in (-2.0, 0.5, 3.0):
        tr = m.run(keep=False, th=th0, v=v0, total=60)
        col = "tab:red" if tr["v"][-100:].min() > 0.1 else "tab:blue"   # still moving at the end
        ax.plot(broken(tr["th"]), tr["v"], color=col, lw=0.7, alpha=0.7)
cyc = m.run(keep=False, th=0, v=2.5, total=200, trans=150)        # the rotating orbit
ax.plot(broken(cyc["th"]), cyc["v"], color="darkred", lw=2.5, label="rotating orbit")
ax.plot(node.x["TH"], 0, "o", color="navy", ms=9, label="stable rest state")
ax.plot(saddle.x["TH"], 0, "o", mfc="white", mec="k", ms=9, label="saddle")
ax.plot([], [], color="tab:blue", label="ends at rest")
ax.plot([], [], color="tab:red", label="ends rotating")
ax.set(xlim=(0, TWO_PI), ylim=(-2.5, 4.9), xlabel="th (folded into [0, 2π))", ylabel="v",
       title=f"Driven pendulum on the cylinder, i = {i0}, b = {b}")
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, TWO_PI], ["0", "π/2", "π", "3π/2", "2π"])
ax.legend(fontsize=8, loc="upper center", ncol=3)
fig.tight_layout()
fig.savefig(FIG / "cylinder.png", dpi=130)

# %% 4. hysteresis: sweep i up, then down, continuing from the last state
I = np.round(np.arange(0, 1.5001, 0.025), 4)
m.run(keep=False, i=0, th=0, v=0, total=10)       # start at rest
up, down = [], []
for sweep, values in ((up, I), (down, I[::-1])):
    for x in values:
        m.set(i=x)
        m.run_last(total=300)                     # settle
        sweep.append(mean_velocity(m.run_last(total=300)))
up, down = np.array(up), np.array(down)[::-1]
m.set(i=i0)
i_leave = I[np.argmax(up > 0)]                    # first i at which the up sweep rotates
i_stop = I[np.argmax(down > 0)]                   # smallest i at which the down sweep still rotates
print(f"\nup sweep starts rotating at i = {i_leave}; down sweep stops below i = {i_stop}")

fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.plot(I, I / b, color="0.6", ls=":", label="i / b")
ax.plot(I, up, "o-", color="tab:blue", ms=4, label="i increasing")
ax.plot(I, down, "s--", color="tab:red", ms=4, mfc="none", label="i decreasing")
ax.axvline(1, color="tab:blue", lw=0.8, ls="--")
ax.text(1.01, 0.2, "rest states end\n(sin th = i)", fontsize=8, color="tab:blue")
ax.set(xlim=(0, 1.5), ylim=(-0.2, 5.3), xlabel="torque i", ylabel="mean velocity",
       title=f"Hysteresis between resting and rotating (b = {b})")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "hysteresis.png", dpi=130)


# %% 5. checks
def rotates(i, bb):
    tr = m.run(keep=False, i=i, b=bb, th=0, v=i / bb + 1, total=600, trans=400, dt=0.02)
    return len(wraps(tr)) >= 2


print("\nchecks:")
check("stable rest state th = arcsin(i) (exact)", node.x["TH"], np.arcsin(i0), 1e-6)
check("saddle th = pi - arcsin(i) (exact)", saddle.x["TH"], np.pi - np.arcsin(i0), 1e-6)
# linearisation at a rest state: lambda^2 + b lambda + cos th = 0, so lambda1 * lambda2 = cos th
check("node: eigenvalue product = cos th (exact)", np.prod(node.eigenvalues).real, np.cos(np.arcsin(i0)), 1e-6)
check("saddle: eigenvalue product = -cos th (exact)", np.prod(saddle.eigenvalues).real,
      -np.cos(np.arcsin(i0)), 1e-6)
check("up sweep leaves rest at i = 1 (grid 0.025)", i_leave, 1.0, 0.025)
# Melnikov: the rotating orbit is born in a homoclinic loop at i = 4b/pi + O(b^2) relative error
bb, lo, hi = 0.05, 0.02, 0.1
for _ in range(18):
    lo, hi = (lo, (lo + hi) / 2) if rotates((lo + hi) / 2, bb) else ((lo + hi) / 2, hi)
check("homoclinic torque / (4b/pi) at b = 0.05 (Melnikov)", hi / (4 * bb / np.pi), 1.0, bb**2)
# over whole turns <v'> = 0, so b<v> = i - <sin th>; for fast rotation th ~ W t and to second
# order <sin th> = b / (2 W (W^2 + b^2)), so <v> = i/b - 1/(2 W (W^2 + b^2)) with W ~ i/b
W = 1.5 / b
check("mean velocity at i = 1.5 vs i/b - 1/(2W(W^2+b^2))", up[-1], W - 1 / (2 * W * (W**2 + b**2)), 2e-4)
check("up and down sweeps agree at i = 1.5", down[-1], up[-1], 1e-4)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
