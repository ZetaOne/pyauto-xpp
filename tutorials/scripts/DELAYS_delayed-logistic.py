"""
Hutchinson's delayed logistic equation
======================================

    x' = r x (1 - x(t - tau))

Logistic growth where crowding acts with a delay tau (for example the time to
mature).  The equilibrium x = 1 is stable while r*tau < pi/2; at r*tau = pi/2 it
loses stability in a Hopf bifurcation and the population oscillates, with
period 4*tau at onset.  The history for t < 0 is given by a formula.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. shows a damped and a sustained oscillation, in time and in the delay
     plane (x(t - tau), x(t))
  3. sweeps r at tau = 1 and tau at r = 1: the amplitude at long times, and
     the onset found by extrapolating amplitude^2 to zero
  4. measures the period just above onset and compares it with 4*tau
  5. checks the results against exact values for this model

AUTO does not handle delay equations here, so everything comes from simulation.

Run:   python DELAYS_delayed-logistic.py
Figures go to figures/DELAYS_delayed-logistic/.

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
    NAME = "DELAYS_delayed-logistic"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


def period(tr):
    """Mean time between upward crossings of x = 1."""
    x, t = tr["x"] - 1, tr.t
    up = t[1:][(x[:-1] < 0) & (x[1:] >= 0)]
    return np.diff(up).mean()


# %% 1. the model
# A small step: XPP's delayed values behave as if the delay were about dt/2
# longer, which moves the onset by about r*dt/(2*tau); with dt = 0.002 that is 0.1%.
# XPP takes x(0) as the leading number of the history formula (0.5), so the formula
# is chosen to equal 0.5 at t = 0 (XPP Studio's 0.5+0.2*cos(3*t) jumps there).
mb = (ModelBuilder("delayed_logistic")
      .comment("Hutchinson's delayed logistic equation")
      .ode("x", "r*x*(1-delay(x,tau))")
      .aux("xlag", "delay(x,tau)")
      .parameter(r=1.8, tau=1)
      .initial_value("x", "0.5+0.2*sin(3*t)")    # history for t <= 0 (see below)
      .option(total=100, dt=0.002, delay=5))     # delay = the largest tau used
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. damped (r = 1.3) and sustained (r = 1.8) oscillations, tau = 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for r, col in ((1.3, "tab:blue"), (1.8, "tab:red")):
    tr = m.run(keep=False, r=r, total=60)
    ax1.plot(tr.t, tr["x"], color=col, lw=1.2, label=f"r = {r}")
    ax2.plot(tr["xlag"], tr["x"], color=col, lw=1)
ax1.axhline(1, color="tab:gray", lw=0.6)
ax1.set(xlabel="t", ylabel="x", title="tau = 1: r tau below and above pi/2")
ax1.legend()
ax2.set(xlabel="x(t - tau)", ylabel="x(t)", title="delay plane")
fig.tight_layout()
fig.savefig(FIG / "time_series.png", dpi=130)

# %% 3. amplitude at long times: against r (tau = 1) and against tau (r = 1)
def amp(r, tau, total=600, trans=500):
    return np.ptp(m.run(keep=False, r=r, tau=tau, total=total, trans=trans)["x"])


# Near a supercritical Hopf point amplitude^2 grows linearly in the distance past it,
# so fit amplitude^2 at a few values just above onset and extrapolate to zero.
# Close to onset the oscillation settles slowly: long runs there.
def onset(values, f):
    a2 = np.array([f(v)**2 for v in values])
    slope, icept = np.polyfit(values, a2, 1)
    return -icept / slope


R = np.linspace(1.0, 2.2, 25)
T = np.linspace(1.0, 2.2, 25)
amp_r = [amp(r, 1) for r in R]
amp_t = [amp(1, t) for t in T]
near = np.linspace(1.58, 1.60, 5)                    # r*tau a little above pi/2
r_c = onset(near, lambda r: amp(r, 1, 1500, 1300))
tau_c = onset(near, lambda t: amp(1, t, 1500, 1300))
print(f"\nonset: r = {r_c:.5f} at tau = 1, tau = {tau_c:.5f} at r = 1 (pi/2 = {np.pi / 2:.5f})")

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, v, a, c, lab in ((axs[0], R, amp_r, r_c, "r  (tau = 1)"), (axs[1], T, amp_t, tau_c, "tau  (r = 1)")):
    ax.plot(v, a, "o-", ms=4)
    ax.axvline(np.pi / 2, color="tab:red", ls="--", lw=1, label="pi/2 (exact)")
    ax.axvline(c, color="k", ls=":", lw=1, label=f"from amplitude^2: {c:.4f}")
    ax.set(xlabel=lab)
    ax.legend(fontsize=8)
axs[0].set_ylabel("peak-to-peak amplitude of x")
fig.suptitle("Hopf bifurcation of the delayed logistic equation")
fig.tight_layout()
fig.savefig(FIG / "amplitude.png", dpi=130)

# %% 4. period near onset, and the r*tau scaling
P_on = period(m.run(keep=False, r=1.58, tau=1, total=1500, trans=1300))
P1 = m.run(keep=False, r=1.8, tau=1, total=300, trans=200)
# the rescaled problem has step dt/tau, so give tau = 2 the step 2*dt: the same number
# of steps per delay (with equal dt, XPP's dt/2 lag would make r*tau_eff differ slightly)
P2 = m.run(keep=False, r=0.9, tau=2, dt=0.004, total=600, trans=400)
print(f"period at r tau = 1.58: {P_on:.4f};  r tau = 1.8: period {period(P1):.4f} (tau = 1), "
      f"{period(P2):.4f} (tau = 2)")

# %% 5. checks
print("\nchecks:")
# exact: x = 1 + u gives u' = -r u(t - tau); with u = e^(i w t), i w = -r e^(-i w tau),
# so cos(w tau) = 0 and w = r: w tau = pi/2, i.e. r tau = pi/2 and period 2 pi / w = 4 tau
check("onset in r at tau = 1 (exact pi/2)", r_c, np.pi / 2, 0.01)
check("onset in tau at r = 1 (exact pi/2)", tau_c, np.pi / 2, 0.01)
# just above onset (r tau = 1.58) the period is still close to its onset value
check("period just above onset, tau = 1 (exact at onset: 4)", P_on, 4.0, 0.05)
# exact: s = t/tau turns the equation into dx/ds = (r tau) x (1 - x(s - 1)), so only
# r*tau matters: the same amplitude, and a period proportional to tau
check("same amplitude for (r, tau) = (1.8, 1) and (0.9, 2)", np.ptp(P2["x"]), np.ptp(P1["x"]), 1e-3)
check("period ratio for tau = 2 and tau = 1 (exact 2)", period(P2) / period(P1), 2.0, 1e-3)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
