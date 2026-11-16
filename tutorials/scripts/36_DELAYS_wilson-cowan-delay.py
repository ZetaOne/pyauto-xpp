"""
Wilson-Cowan with a delay: an excitatory-inhibitory pair starts to oscillate
============================================================================

    E' = -E + S(6 E - 8 I(t-tau) + P)
    I' = (-I + S(8 E(t-tau) - I + Q)) / 0.5,        S(u) = 1/(1 + exp(-u))

An excitatory and an inhibitory population that reach each other only after a
delay tau.  For Q = 0 there is one equilibrium for every input P in [-1, 4].
With tau = 1 it loses stability at P = 1.54359, where a pair of characteristic
roots crosses at +-0.87492 i, and a stable rhythm appears.  Shorter delays
need a stronger input: the Hopf curve in (P, tau) bends to larger P as tau
falls.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. follows the equilibrium in P with AUTO (tau = 1) and finds the Hopf point
  3. follows the Hopf point in (P, tau) and checks every point against the
     characteristic equation, and the curve against four verified points
  4. continues the periodic orbits from the Hopf point and compares them with
     simulations on both sides of it

Run:   python 36_DELAYS_wilson-cowan-delay.py
Figures go to figures/36_DELAYS_wilson-cowan-delay/.

Model from the delay-equation examples accompanying XPPAUT 8.2.
Every number is computed by XPPAUT 8.2 and its AUTO through pyxpp.
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
    NAME = "36_DELAYS_wilson-cowan-delay"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.7g} (expected {expected:.7g})")


# %% 1. the model
# The history (t < 0) is XPP's E(0)= / I(0)= formula: here the parameters e0, i0.
mb = (ModelBuilder("wilson_cowan_delay")
      .comment("Wilson-Cowan E-I pair with delayed E<->I interaction")
      .function("s(u)", "1/(1+exp(-u))")
      .ode("E", "-E+s(6*E-8*delay(I,tau)+P)")
      .ode("I", "(-I+s(8*delay(E,tau)-I+Q))/0.5")
      .parameter(P=0.0, Q=0.0, tau=1.0, e0=0.031767, i0=0.450954)
      .initial_value("E", "e0")
      .initial_value("I", "i0")
      .initial(E=0.031767, I=0.450954)          # the equilibrium at P = 0
      .option(delay=2, meth="rk4", total=200, dt=0.01, bound=1000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")
m.auto.parameters("P", "Q", "tau")


def char_det(P, Q, E, I, tau, w):
    """det Delta(i w) of the linearization at (E, I), relative to its size:
    Delta(l) = [[l + 1 - 6 g1, 8 g1 exp(-l tau)], [-16 g2 exp(-l tau), l + 2 + 2 g2]],
    g1 = S'(6E - 8I + P), g2 = S'(8E - I + Q), S' = S (1 - S)"""
    s = lambda u: 1 / (1 + np.exp(-u))                                   # noqa: E731
    g1 = s(6 * E - 8 * I + P) * (1 - s(6 * E - 8 * I + P))
    g2 = s(8 * E - I + Q) * (1 - s(8 * E - I + Q))
    l, z = 1j * w, np.exp(-2j * w * tau)
    d1, d2 = l + 1 - 6 * g1, l + 2 + 2 * g2
    return abs(d1 * d2 + 128 * g1 * g2 * z) / (abs(d1 * d2) + 128 * g1 * g2)


# %% 2. the equilibrium in P, tau = 1
m.run(total=1)
m.run_last(total=1)
eqs = m.auto.run("P", start="steady", ds=0.01, dsmax=0.005, parmin=-1, parmax=4, nmax=3000)
hb = eqs.special("HB")[0]
print(f"\nHopf at P = {hb.x:.6f}, omega = {2 * np.pi / hb.period:.6f}")

# %% 3. the Hopf curve in (P, tau)
# epsl/epsu: AUTO's Newton tolerances (default 1e-4 is too loose for these checks)
# dsmax=0.01: points close enough together to read P and omega off the curve by interpolation
up = m.auto.run(grab=hb.label, choice="two param", param2="tau", ds=0.01, dsmax=0.01, nmax=400,
                epsl=1e-8, epsu=1e-8, parmin=-1, parmax=4)
dn = m.auto.run(grab=hb.label, choice="two param", param2="tau", ds=-0.01, dsmax=0.05, nmax=120,
                epsl=1e-8, epsu=1e-8, parmin=-1, parmax=4)
curve = sorted(list(up) + list(dn), key=lambda p: p.par["tau"])
cP = np.array([p.x for p in curve])
ctau = np.array([p.par["tau"] for p in curve])
cw = np.array([2 * np.pi / p.period for p in curve])
res = [char_det(p.x, 0, p.u0["E"], p.u0["I"], p.par["tau"], 2 * np.pi / p.period) for p in curve]
print(f"Hopf curve: {len(curve)} points, tau from {ctau.min():.3f} to {ctau.max():.2f}")
m.set(P=0.0, tau=1.0)                     # AUTO leaves the last point's parameters behind

# %% 4. periodic orbits and simulations
po = m.auto.run(grab=hb.label, choice="periodic", ds=0.01, dsmax=0.05, nmax=200, parmin=-1, parmax=4)
pP = np.array([p.x for p in po])
pmax = np.array([p.umax["E"] for p in po])
pmin = np.array([p.umin["E"] for p in po])
pper = np.array([p.period for p in po])
print(f"{len(po)} periodic orbits from P = {pP[0]:.4f}, stable at {sum(p.stable for p in po)}")
sims = {}
for P in (1.3, 2.0, 3.0):
    tr = m.run(keep=False, P=P, tau=1.0, E=0.2, e0=0.2, I=0.5, i0=0.5, total=400, trans=250)
    t, e = tr.t, tr["E"]
    lev = 0.5 * (e.max() + e.min())
    upc = np.where((e[:-1] < lev) & (e[1:] >= lev))[0]
    tc = t[upc] + (lev - e[upc]) / (e[upc + 1] - e[upc]) * (t[upc + 1] - t[upc])
    sims[P] = (tr, e.max(), e.min(), np.diff(tc).mean() if len(tc) > 2 else np.nan)

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
xs = np.array([p.x for p in eqs])
st = np.array([p.stable for p in eqs])
ee = np.array([p.u0["E"] for p in eqs])
axs[0].plot(xs, np.where(st, ee, np.nan), "r-", lw=2.5, label="stable equilibrium")
axs[0].plot(xs, np.where(~st, ee, np.nan), "k-", lw=1, label="unstable equilibrium")
axs[0].plot(pP, pmax, "g-", lw=2, label="AUTO orbits: max and min E")
axs[0].plot(pP, pmin, "g-", lw=2)
for P, (tr, smx, smn, _) in sims.items():
    axs[0].plot([P, P], [smx, smn], "ko", mfc="none", ms=7, label="simulations" if P == 1.3 else None)
axs[0].plot(hb.x, hb.u0["E"], "o", color="purple", ms=8, label="Hopf")
axs[0].set(xlim=(0, 4), xlabel="P", ylabel="E", title="tau = 1")
axs[0].legend(fontsize=7, loc="upper left")
axs[1].plot(cP, ctau, "k-", lw=1.5, label="Hopf curve (AUTO)")
ref = np.array([(1.5436, 1.0000, 0.8749), (1.7306, 0.8200, 0.9869), (2.3153, 0.5157, 1.2717),
                (3.1739, 0.3400, 1.5005)])
axs[1].plot(ref[:, 0], ref[:, 1], "o", color="tab:orange", mfc="none", ms=9, label="verified points")
axs[1].set(xlim=(0.5, 4), ylim=(0, 4), xlabel="P", ylabel="tau", title="Hopf curve in (P, tau)")
axs[1].text(2.6, 2.0, "oscillations", fontsize=9)
axs[1].text(0.7, 0.6, "rest", fontsize=9)
axs[1].legend(fontsize=8)
for P, col in ((1.3, "tab:blue"), (2.0, "tab:red")):
    tr = m.run(keep=False, P=P, tau=1.0, E=0.2, e0=0.2, I=0.5, i0=0.5, total=60)
    axs[2].plot(tr.t, tr["E"], color=col, lw=1.2, label=f"P = {P}")
axs[2].set(xlabel="t", ylabel="E", title="Simulations, tau = 1")
axs[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "bifurcation.png", dpi=130)

# %% 5. checks
print("\nchecks:")
check("Hopf point P at tau = 1", hb.x, 1.54359, 2e-5)
check("  omega there", 2 * np.pi / hb.period, 0.87492, 2e-5)
check("  |det Delta(i omega)| there (relative)", char_det(hb.x, 0, hb.u0["E"], hb.u0["I"], 1, 2 * np.pi / hb.period), 0, 1e-5)
check(f"Hopf curve: largest relative |det Delta(i omega)| on {len(curve)} points", max(res), 0, 1e-5)
# the verified points are rounded to 4 decimals; read the curve at the listed P (tau changes
# slowly with P where the curve is steep in P, so this is the well-conditioned direction)
o = np.argsort(cP)
for P_r, tau_r, w_r in ref[1:]:
    check(f"curve at P = {P_r}: tau", np.interp(P_r, cP[o], ctau[o]), tau_r, 1e-4)
    check(f"curve at P = {P_r}: omega", np.interp(P_r, cP[o], cw[o]), w_r, 1e-4)
check("orbits stable (1 = yes)", float(all(p.stable for p in po)), 1, 0)
check("P = 1.3: simulation settles (max - min of E)", sims[1.3][1] - sims[1.3][2], 0, 1e-5)
for P in (2.0, 3.0):
    _, smx, smn, sper = sims[P]
    check(f"P = {P}: max E, AUTO vs simulation", np.interp(P, pP, pmax), smx, 2e-3)
    check(f"P = {P}: period, AUTO vs simulation", np.interp(P, pP, pper), sper, 1e-2)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
