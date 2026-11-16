"""
Integrate-and-fire neuron with adaptation
=========================================

The voltage v charges towards the input i with time constant tau.  When it
reaches the threshold 1 a `global` event resets it to 0 and kicks an adaptation
current w up by b; w then decays with the slower time constant tauw.  Each spike
makes the next one come later, until the intervals settle at a steady rhythm.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. plots v and w and the inter-spike intervals growing to a steady value
  3. computes the steady interval exactly (from the periodic solution between
     resets) and compares it with XPP's
  4. without adaptation (b = 0), compares the interval with tau*ln(i/(i-1))
  5. draws the f-I curve with and without adaptation; both start at i = 1
  6. checks the results against those exact values

Run:   python EVENTS_integrate-and-fire.py
Figures go to figures/EVENTS_integrate-and-fire/.

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
    NAME = "EVENTS_integrate-and-fire"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
tau, tauw, b, i0 = 1.0, 20.0, 0.2, 1.6
mb = (ModelBuilder("integrate_and_fire")
      .comment("leaky integrate-and-fire neuron with spike-triggered adaptation")
      .ode("v", "(-v+i-w)/tau")
      .ode("w", "-w/tauw")
      .global_event(1, "v-1", {"v": 0, "w": "w+b"})     # threshold 1: reset and adapt
      .parameter(i=i0, tau=tau, tauw=tauw, b=b)
      .initial(v=0, w=0)
      .option(total=200, dt=0.001, maxstor=300000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP


def spike_times(tr, i):
    """reset times: the first output point after each reset, moved back by the time
    v needs to climb from 0 to its value there (v' = (i - w)/tau just after a reset)"""
    k = np.where(np.diff(tr["v"]) < -0.5)[0] + 1
    return tr.t[k] - tau * tr["v"][k] / (i - tr["w"][k])


# %% 2. adaptation: intervals lengthen and settle
tr = m.run()
sp = spike_times(tr, i0)
isi = np.diff(sp)
print(f"\n{len(sp)} spikes; first intervals", np.round(isi[:5], 4), "... last", np.round(isi[-1], 5))

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})
sel = tr.t <= 60
axs[0].plot(tr.t[sel], tr["v"][sel], lw=0.9, label="v")
axs[0].plot(tr.t[sel], tr["w"][sel], lw=1.5, label="w (adaptation)")
axs[0].axhline(1, color="k", ls=":", lw=0.8)
axs[0].set(xlabel="t", title=f"Spiking with adaptation, i = {i0}, b = {b}")
axs[0].legend(fontsize=8, loc="center right")


# exact steady state: just after a reset w = W, and W = W exp(-T/tauw) + b, so
# W = b/(1 - exp(-T/tauw)).  Solving v' = (-v + i - W exp(-t/tauw))/tau from v(0) = 0,
#   v(t) = i (1 - exp(-t/tau)) - W tauw/(tauw - tau) (exp(-t/tauw) - exp(-t/tau)),
# and the interval T is the root of v(T) = 1 (found by bisection below).
def steady_isi(i, b):
    def f(T):
        W = b / (1 - np.exp(-T / tauw)) if b > 0 else 0.0
        return i * (1 - np.exp(-T / tau)) - W * tauw / (tauw - tau) * (np.exp(-T / tauw) - np.exp(-T / tau)) - 1
    lo, hi = 1e-6, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
    return 0.5 * (lo + hi)


T_ad = steady_isi(i0, b)
axs[1].plot(np.arange(1, len(isi) + 1), isi, "o", ms=4, label="XPP")
axs[1].axhline(T_ad, color="tab:red", lw=1, label=f"exact steady interval {T_ad:.4f}")
axs[1].axhline(tau * np.log(i0 / (i0 - 1)), color="tab:gray", ls="--", lw=1, label="no adaptation")
axs[1].set(xlabel="spike number", ylabel="inter-spike interval", title="Intervals settle")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "adaptation.png", dpi=130)

# %% 3. no adaptation: v(t) = i (1 - exp(-t/tau)) reaches 1 at T = tau ln(i/(i-1))
tr0 = m.run(keep=False, b=0, total=40)
isi0 = np.diff(spike_times(tr0, i0))
print(f"b = 0: intervals {np.round(isi0[:4], 6)}, exact {tau * np.log(i0 / (i0 - 1)):.6f}")

# %% 4. f-I curves
I = np.round(np.concatenate([np.arange(0.9, 1.1, 0.01), np.linspace(1.1, 4, 30)]), 10)
rates = {}
for bb in (0.0, b):
    r = []
    for i in I:
        # near i = 1 the adapted interval is long (about 62 at i = 1.01): wait 400, look 400
        t_ = spike_times(m.run(keep=False, i=i, b=bb, total=800, trans=400, dt=0.002), i)
        r.append(1 / np.mean(np.diff(t_)) if len(t_) > 1 else 0.0)
    rates[bb] = np.array(r)

Ifine = np.linspace(1.0001, 4, 400)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Ifine, 1 / (tau * np.log(Ifine / (Ifine - 1))), "k-", lw=0.8, label="1/(tau ln(i/(i-1)))")
ax.plot(Ifine, [1 / steady_isi(i, b) for i in Ifine], "k--", lw=0.8, label="exact, with adaptation")
ax.plot(I, rates[0.0], "o", ms=5, label="XPP, b = 0")
ax.plot(I, rates[b], "s", ms=5, label=f"XPP, b = {b}")
ax.axvline(1, color="tab:gray", ls=":", lw=1)
ax.set(xlabel="input i", ylabel="steady firing rate", title="f-I curves: rheobase i = 1")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "f_I.png", dpi=130)

# %% 5. checks
print("\nchecks:")
check("interval without adaptation, tau ln(i/(i-1)) (exact)", np.mean(isi0), tau * np.log(i0 / (i0 - 1)), 1e-5)
check("steady interval with adaptation (exact)", isi[-1], T_ad, 1e-4)
# rheobase: for i <= 1 the rest state v = i is below threshold, for any i > 1 the cell fires;
# adaptation cannot move it because w decays to 0 while the cell is silent
for bb in (0.0, b):
    check(f"rheobase, b = {bb}: largest silent i on the 0.01 grid (exact 1)", I[rates[bb] == 0].max(), 1.0, 1e-9)
    check(f"rheobase, b = {bb}: smallest firing i on the 0.01 grid (exact 1.01)", I[rates[bb] > 0].min(), 1.01, 1e-9)
k = np.argmin(abs(I - 3.0))
check("rate at i = 3, b = 0: 1/ln(3/2) (exact)", rates[0.0][k], 1 / np.log(I[k] / (I[k] - 1)), 1e-4)
check("rate at i = 3, adapted (exact)", rates[b][k], 1 / steady_isi(I[k], b), 1e-4)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
