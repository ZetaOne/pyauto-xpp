"""
Bouncing ball: global events
============================

A ball dropped from height h falls under gravity, x' = v, v' = -g.  A `global`
event fires whenever x crosses 0 going down: it puts the ball back on the floor
and reverses its velocity with a coefficient of restitution e < 1.  Each flight
is shorter than the one before by the factor e, so infinitely many bounces fit
into a finite time (a Zeno point), after which the ball lies still.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. runs it and plots height and velocity against time, and the phase plane
  3. reads the bounce times from XPP's output and recovers each impact time
     exactly from the ballistic motion between output points
  4. compares flight times, energies and the time to come to rest with the
     closed-form answers
  5. shows that XPP's event times converge like dt^2
  6. checks the results against those exact values

Run:   python 03_EVENTS_bouncing-ball.py
Figures go to figures/03_EVENTS_bouncing-ball/.

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
    NAME = "03_EVENTS_bouncing-ball"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
g, e, h = 9.81, 0.85, 2.0
mb = (ModelBuilder("bouncing_ball")
      .comment("ball bouncing on the floor; the global event is the impact")
      .ode("x", "v")
      .ode("v", "-g")
      .global_event(-1, "x", {"x": 0, "v": "-e*v"})    # x crosses 0 downwards
      .parameter(g=g, e=e)
      .initial(x=h, v=0)
      .option(total=9, dt=0.0002))       # small dt: see section 4
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. a run
tr = m.run()
t, x, v = tr.t, tr["x"], tr["v"]

# exact: first fall lasts t1 = sqrt(2h/g); flight n (after bounce n) lasts 2 e^n t1,
# so the ball is at rest at T = t1 + 2 t1 (e + e^2 + ...) = t1 (1+e)/(1-e)
t1 = np.sqrt(2 * h / g)
T_rest = t1 * (1 + e) / (1 - e)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
axs[0].plot(t, x, lw=1.2, label="height x")
axs[0].plot(t, v / 4, lw=0.8, color="tab:gray", label="velocity v / 4")
axs[0].axvline(T_rest, color="tab:red", ls="--", lw=1, label=f"rest at t1(1+e)/(1-e) = {T_rest:.3f}")
axs[0].axhline(0, color="k", lw=0.5)
axs[0].annotate("past T the flights are shorter than dt,\nthe event is missed and the\nball falls through the floor",
                xy=(8.25, -0.9), xytext=(4.6, -1.65), fontsize=8, arrowprops=dict(arrowstyle="->"))
axs[0].set(xlim=(0, 8.8), ylim=(-1.9, 2.2), xlabel="t", title=f"Bouncing ball, e = {e}")
axs[0].legend(fontsize=8, loc="upper right")
axs[1].plot(x, v, lw=0.8)
axs[1].set(xlim=(-0.1, 2.1), ylim=(-6.6, 6), xlabel="x", ylabel="v", title="Phase plane: the jumps are the impacts")
fig.tight_layout()
fig.savefig(FIG / "bounces.png", dpi=130)

# %% 3. impact times
# XPP stops the step at the crossing, applies the jump, and writes the next output point.
# Between impacts the motion is exactly ballistic, so from the first output point after
# an impact, (x_k, v_k) at t_k, the time s since the impact solves x_k = v_k s + g s^2/2.
k = np.where(np.diff(v) > 0)[0] + 1                  # first output point after each jump
k = k[t[k] < T_rest - 0.1]                           # leave the unresolvable last few bounces out
s = (-v[k] + np.sqrt(v[k] ** 2 + 2 * g * x[k])) / g
impacts = t[k] - s
flights = np.diff(impacts)
energy = 0.5 * v[k] ** 2 + g * x[k]                  # energy per unit mass in each flight
print(f"\n{len(impacts)} impacts resolved; first ones at", np.round(impacts[:6], 5))
print("flight ratios:", np.round(flights[1:8] / flights[:7], 5))
print("energy ratios:", np.round(energy[1:8] / energy[:7], 5), f"(e^2 = {e**2:.4f})")

n = np.arange(1, len(impacts) + 1)
exact = t1 * (1 + e - 2 * e ** n) / (1 - e)          # t_n = t1 + 2 t1 (e + ... + e^(n-1))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].plot(n, impacts, "o", label="XPP")
axs[0].plot(n, exact, "k-", lw=1, label="t1 (1 + e - 2 e^n)/(1 - e)")
axs[0].axhline(T_rest, color="tab:red", ls="--", lw=1, label="Zeno time")
axs[0].set(xlabel="impact n", ylabel="time of impact", title="Impacts pile up before T")
axs[0].legend(fontsize=8)
axs[1].semilogy(n[:-1], flights, "o", label="flight time")
axs[1].semilogy(n, energy, "s", label="energy")
axs[1].semilogy(n, 2 * t1 * e ** n, "k-", lw=0.8, label="2 t1 e^n")
axs[1].semilogy(n, g * h * e ** (2 * n), "k--", lw=0.8, label="g h e^(2n)")
axs[1].set(xlabel="impact n", title="Geometric decay")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "impacts.png", dpi=130)

# the simulation's own estimate of the rest time: last impact plus the geometric tail
# of the remaining flights, flights[-1] * (e + e^2 + ...) = flights[-1] * e/(1-e)
T_sim = impacts[-1] + flights[-1] * e / (1 - e)

# %% 4. how accurate is XPP's event location?
# XPP finds the crossing inside a step by interpolation, so impact times carry an error
# that shrinks like dt^2: five times smaller dt, about 25 times smaller error,
# which is why this script uses dt = 0.0002.
print("\nerror of the 10th impact time against the exact value:")
for dt in (0.01, 0.002, 0.0004):
    r = m.run(keep=False, dt=dt)
    kk = np.where(np.diff(r["v"]) > 0)[0][9] + 1
    ss = (-r["v"][kk] + np.sqrt(r["v"][kk] ** 2 + 2 * g * r["x"][kk])) / g
    print(f"  dt = {dt:<7} error = {r.t[kk] - ss - exact[9]:+.2e}")

# %% 5. checks
print("\nchecks:")
check("first impact at sqrt(2h/g) (exact)", impacts[0], t1, 1e-6)
check("flight time ratio, mean of 10 (exact e)", np.mean(flights[1:11] / flights[:10]), e, 1e-5)
check("energy ratio per bounce, mean of 10 (exact e^2)", np.mean(energy[1:11] / energy[:10]), e**2, 1e-5)
check("tenth impact time (exact)", impacts[9], exact[9], 1e-5)
check("time to rest from the impacts vs t1(1+e)/(1-e)", T_sim, T_rest, 1e-4)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
