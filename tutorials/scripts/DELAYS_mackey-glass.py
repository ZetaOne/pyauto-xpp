"""
Mackey-Glass: chaos from a delay
================================

    x' = b x(t - tau) / (1 + x(t - tau)^pw) - g x

A model of blood-cell production: cells are made at a rate set by the
population tau time units ago and die at rate g.  For small tau the population
settles to a steady level; as tau grows it oscillates, the oscillation doubles
its period, and at tau = 17 the dynamics are chaotic, although the equation has
only one variable.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. checks the steady level x* from long runs at small tau
  3. sweeps tau and plots the maxima of x after transients
  4. draws the delay embedding (x(t - tau), x(t)) at four values of tau
  5. finds where the steady level loses stability, from the amplitude just
     past onset, and the period there
  6. checks the results against exact values for this model

AUTO does not handle delay equations here, so everything comes from simulation.

Run:   python DELAYS_mackey-glass.py
Figures go to figures/DELAYS_mackey-glass/.

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


def maxima(x):
    return x[np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1]


# %% 1. the model
b, g, pw = 0.2, 0.1, 10
mb = (ModelBuilder("mackey_glass")
      .comment("Mackey-Glass equation")
      .ode("x", "b*delay(x,tau)/(1+delay(x,tau)^pw)-g*x")
      .aux("xlag", "delay(x,tau)")
      .parameter(b=b, g=g, pw=pw, tau=17)
      .initial(x=0.5)                     # constant history x = 0.5 for t <= 0
      .option(total=1000, dt=0.01, delay=40))  # delay = the largest tau used, with room
# XPP sizes its store of past values from dt and the maximal delay when the model
# loads, and pyxpp does not resize it when dt is changed later.  So the model gets
# the smallest step used (0.01, for section 5); the sweeps run with the larger
# step DT, which is safe (a smaller dt than at load time is not).
DT = 0.1
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. the steady level at small tau
x_end = {tau: m.run(keep=False, tau=tau, dt=DT, total=2000)["x"][-1] for tau in (1, 2, 4)}
print("\nx at t = 2000:", {k: round(v, 7) for k, v in x_end.items()})

# %% 3. maxima of x against tau
TAU, M = [], []
for tau in np.linspace(4, 25, 211):
    mx = maxima(m.run(keep=False, tau=tau, dt=DT, total=3000, trans=2000)["x"])
    TAU += [tau] * len(mx)
    M += list(mx)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(TAU, M, "k.", ms=1)
ax.axhline(1, color="tab:gray", lw=0.8, ls=":", label="steady level x* = 1")
ax.set(xlim=(4, 25), xlabel="tau", ylabel="maxima of x", title="Mackey-Glass: maxima of x against the delay")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "maxima_vs_tau.png", dpi=130)

# %% 4. delay embedding
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True, sharey=True)
for ax, tau, lab in zip(axs, (8, 14, 15.8, 17), ("period 1", "period 2", "period 4", "chaos")):
    tr = m.run(keep=False, tau=tau, dt=DT, total=3000, trans=2000)
    ax.plot(tr["xlag"], tr["x"], lw=0.5)
    ax.set(title=f"tau = {tau}: {lab}", xlabel="x(t - tau)")
axs[0].set_ylabel("x(t)")
fig.tight_layout()
fig.savefig(FIG / "embedding.png", dpi=130)

# %% 5. loss of stability of the steady level
# Amplitude^2 grows linearly just past a supercritical Hopf point: fit and
# extrapolate to zero.  A finer step here: XPP's delayed values lag by about dt/2,
# which shifts the onset by about dt/2 (0.005 with dt = 0.01).
near = np.linspace(4.75, 4.95, 5)
a2 = [np.ptp(m.run(keep=False, tau=t, dt=0.01, total=4000, trans=3800)["x"])**2 for t in near]
slope, icept = np.polyfit(near, a2, 1)
tau_c = -icept / slope
tr = m.run(keep=False, tau=4.75, dt=0.01, total=4000, trans=3000)
x, t = tr["x"] - 1, tr.t
up = t[1:][(x[:-1] < 0) & (x[1:] >= 0)]
P_on = np.diff(up).mean()
print(f"\nsteady level loses stability at tau = {tau_c:.4f}; period at tau = 4.75: {P_on:.3f}")

# %% 6. checks
print("\nchecks:")
# exact: x' = 0 with x(t - tau) = x gives b/(1 + x^pw) = g, x* = (b/g - 1)^(1/pw) = 1
x_star = (b / g - 1) ** (1 / pw)
for tau, xe in x_end.items():
    check(f"steady level at tau = {tau} (exact)", xe, x_star, 1e-6)
# exact: linearise about x*: u' = -g u + f' u(t - tau) with f' = b (1 - (pw-1) x*^pw)/(1 + x*^pw)^2
# = -0.4.  Put u = e^(i w t): i w = -g + f' e^(-i w tau), so
#   cos(w tau) = g/f' = -1/4  and  w = -f' sin(w tau) = 0.4 sqrt(15/16),
# and tau_c = arccos(-1/4)/w, period 2 pi/w at onset
fp = b * (1 - (pw - 1) * x_star**pw) / (1 + x_star**pw) ** 2
w = np.sqrt(fp**2 - g**2)
tau_exact = np.arccos(g / fp) / w
check("onset of oscillation, tau_c (exact)", tau_c, tau_exact, 0.02)
check("period just past onset (exact at onset: 2 pi/w)", P_on, 2 * np.pi / w, 0.2)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
