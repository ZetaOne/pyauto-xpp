"""
Distributed delay: a Volterra integral equation
===============================================

    x' = r x (1 - int_0^t K(t - s) x(s) ds),   K(u) = u e^(-u/tau) / tau^2

Logistic growth regulated by the population's whole past, weighted by a gamma
kernel with mean delay 2*tau.  XPP writes the integral as int{K#x} and solves the
equation by its Volterra method.  The same kernel can be removed by the linear
chain trick: y1' = (x - y1)/tau, y2' = (y1 - y2)/tau gives y2 = int K x, so the
equation is also a 3-variable ODE.  The equilibrium x = 1 loses stability at
r*tau = 2.

This script
  1. builds the Volterra model and the equivalent ODE with ModelBuilder
  2. compares the two solutions, damped (tau = 1.5) and oscillating (tau = 2.5),
     and shows that their difference shrinks like dt^2
  3. sweeps tau at r = 1 and finds the onset of oscillation from the amplitude
  4. checks the results against exact values for this model

Run:   python DELAYS_distributed-delay-volterra.py
Figures go to figures/DELAYS_distributed-delay-volterra/.

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
from pyxpp import load
from pyxpp.builder import ModelBuilder

try:
    HERE = Path(__file__).resolve().parent
    NAME = Path(__file__).stem
except NameError:            # run as a notebook in Jupyter: the kernel starts in this folder
    HERE = Path.cwd()
    NAME = "DELAYS_distributed-delay-volterra"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the two models
# XPP tabulates the kernel K on the time grid when the model loads.  pyxpp does
# not recompute that table when tau or dt is changed later, so each Volterra
# run with another tau or dt gets its own freshly loaded model.
def volterra(tau, dt=0.05):
    return (ModelBuilder("distributed_delay")
            .comment("logistic growth with a gamma-distributed delay")
            .ode("x", "r*x*(1-int{t*exp(-t/tau)/tau^2#x})")
            .parameter(r=1, tau=float(tau))     # a plain float: the builder writes
                                                #   numpy floats as "np.float64(...)"
            .initial(x=0.5)
            .option(total=200, dt=dt, vmaxpts=round(50 / dt)))   # keep 50 time units of past


mb = volterra(2.5)
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

chain = (ModelBuilder("distributed_delay_chain")
         .comment("the same equation by the linear chain trick: y2 = int K x")
         .ode("x", "r*x*(1-y2)")
         .ode("y1", "(x-y1)/tau")
         .ode("y2", "(y1-y2)/tau")
         .parameter(r=1, tau=2.5)
         .initial(x=0.5, y1=0, y2=0)      # int over 0..t starts at 0
         .option(total=200, dt=0.05)
         .load(HERE / "odes" / f"{NAME}-chain.ode"))

# %% 2. Volterra against the chain ODE
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
diff = {}
for tau, col in ((1.5, "tab:blue"), (2.5, "tab:red")):
    for dt in (0.05, 0.025):
        ode = chain.run(keep=False, tau=tau, dt=dt, total=100)   # the ODE on the same time grid
        mv = m if (tau, dt) == (2.5, 0.05) else load(text=volterra(tau, dt).render())
        vol = mv.run(keep=False, total=100)
        n = min(len(vol), len(ode))
        diff[tau, dt] = np.abs(vol["x"][:n] - ode["x"][:n]).max()
        if mv is not m:
            mv.close()
        if dt == 0.05:
            axs[0].plot(vol.t, vol["x"], color=col, lw=2.5, alpha=0.5, label=f"Volterra, tau = {tau}")
            axs[1].semilogy(vol.t, np.abs(vol["x"] - ode["x"]) + 1e-9, color=col, lw=0.8,
                            label=f"tau = {tau}")
            axs[0].plot(ode.t, ode["x"], "k--", lw=0.8)
    print(f"tau = {tau}: largest difference {diff[tau, 0.05]:.2e} (dt = 0.05), "
          f"{diff[tau, 0.025]:.2e} (dt = 0.025)")
axs[0].plot([], [], "k--", lw=0.8, label="chain ODE")
axs[0].set(ylabel="x", title="Volterra equation and its chain ODE, r = 1")
axs[0].legend(fontsize=8, ncol=3)
axs[1].set(xlabel="t", ylabel="|difference|  (dt = 0.05)", ylim=(1e-8, 1))
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "volterra_vs_chain.png", dpi=130)

# %% 3. onset of oscillation in tau at r = 1
def amp_volterra(tau, total=600, trans=500):
    mv = load(text=volterra(tau).render())
    a = np.ptp(mv.run(total=total, trans=trans)["x"])
    mv.close()
    return a


def amp_chain(tau, total=600, trans=500):
    return np.ptp(chain.run(keep=False, tau=tau, total=total, trans=trans)["x"])


TAU = np.linspace(1.4, 3.0, 17)
A_vol = [amp_volterra(t) for t in TAU]
A_ode = [amp_chain(t) for t in TAU]
# amplitude^2 grows linearly just past a supercritical Hopf point: extrapolate to zero
near = np.linspace(2.02, 2.10, 5)
onsets = {}
for nm, f in (("Volterra", amp_volterra), ("chain ODE", amp_chain)):
    a2 = [f(t, 1500, 1300)**2 for t in near]
    slope, icept = np.polyfit(near, a2, 1)
    onsets[nm] = -icept / slope
tr = chain.run(keep=False, tau=2.02, total=1500, trans=1000)
x, t = tr["x"] - 1, tr.t
P_on = np.diff(t[1:][(x[:-1] < 0) & (x[1:] >= 0)]).mean()
print("\nonset in tau:", {k: round(v, 4) for k, v in onsets.items()}, f"; period at tau = 2.02: {P_on:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(TAU, A_vol, "o", ms=7, mfc="none", label="Volterra (int{K#x})")
ax.plot(TAU, A_ode, "k.-", lw=1, label="chain ODE")
ax.axvline(2, color="tab:red", ls="--", lw=1, label="r tau = 2 (exact)")
ax.set(xlabel="tau  (r = 1)", ylabel="peak-to-peak amplitude of x", title="Onset of oscillation")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "amplitude.png", dpi=130)

# %% 4. checks
print("\nchecks:")
# Both are second-order methods solving the same equation, so the difference
# should drop by 2^2 = 4 when dt is halved
for tau in (1.5, 2.5):
    check(f"Volterra - ODE difference ratio, dt halved, tau = {tau} (2nd order: 4)",
          diff[tau, 0.05] / diff[tau, 0.025], 4.0, 0.4)
# exact: about x = 1, u' = -r v2 with the chain v1' = (u - v1)/tau, v2' = (v1 - v2)/tau,
# so l (1 + l tau)^2 = -r.  With l = i w: imaginary part w (1 - w^2 tau^2) = 0, w tau = 1;
# real part -2 w^2 tau = -r, so r tau = 2, and the period at onset is 2 pi tau
check("onset tau from the Volterra equation (exact 2)", onsets["Volterra"], 2.0, 0.02)
check("onset tau from the chain ODE (exact 2)", onsets["chain ODE"], 2.0, 0.02)
check("period just above onset (exact at onset: 2 pi tau = 4 pi)", P_on, 4 * np.pi, 0.2)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
