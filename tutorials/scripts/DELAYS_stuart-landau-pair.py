"""
Two delay-coupled Stuart-Landau oscillators: torus bifurcations of the in-phase wave
===================================================================================

    z1' = (1 + i om) z1 - |z1|^2 z1 + kappa (z2(t-tau) - z1)
    z2' = (1 + i om) z2 - |z2|^2 z2 + kappa (z1(t-tau) - z2)

Two identical oscillators that see each other with a delay tau.  Both can
oscillate in phase on the same circle, z1 = z2 = r exp(i Omega t), with

    Omega = om - kappa sin(Omega tau),    r^2 = 1 - kappa + kappa cos(Omega tau).

With kappa = 1, om = 1.5 this in-phase wave has period 7.955487 at tau = 1 and
loses stability in torus (Neimark-Sacker) bifurcations at tau = 1.584 and 2.329:
a pair of Floquet multipliers leaves the unit circle and a second frequency
appears.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. simulates one period of the in-phase wave at tau = 1, continues it in tau
     with XPP 8.2's periodic-orbit solver for delay equations and finds the two
     torus points (TR) from the Floquet multipliers
  3. checks the branch against the exact formulas and the torus points in a
     rotating frame, where the wave is an equilibrium whose characteristic
     roots (XPP's Sing pts) cross the imaginary axis there
  4. follows the first torus point in (tau, kappa) up to kappa = 1.2, and checks
     the end of that curve by continuing the wave in tau at kappa = 1.2
  5. simulates on both sides of the first torus point

Run:   python DELAYS_stuart-landau-pair.py      (about 40 s: the torus curve)
Figures go to figures/DELAYS_stuart-landau-pair/.

Model from the delay-equation examples accompanying XPPAUT 8.2.
Every number is computed by XPPAUT 8.2 and its AUTO through pyxpp.
"""
# %% setup
import re
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
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.7g} (expected {expected:.7g})")


OM = 1.5

# %% 1. the model
# history (t < 0) = XPP's x1(0)=, x2(0)= formulas: the parameters h1, h2 (y1, y2 start at 0)
mb = (ModelBuilder("stuart_landau_pair")
      .comment("two delay-coupled Stuart-Landau oscillators")
      .ode("x1", "x1-om*y1-(x1^2+y1^2)*x1+kappa*(delay(x2,tau)-x1)")
      .ode("y1", "om*x1+y1-(x1^2+y1^2)*y1+kappa*(delay(y2,tau)-y1)")
      .ode("x2", "x2-om*y2-(x2^2+y2^2)*x2+kappa*(delay(x1,tau)-x2)")
      .ode("y2", "om*x2+y2-(x2^2+y2^2)*y2+kappa*(delay(y1,tau)-y2)")
      .parameter(tau=1, kappa=1, om=OM, h1=1, h2=1)
      .initial_value("x1", "h1")
      .initial_value("x2", "h2")
      .initial(x1=1, y1=0, x2=1, y2=0)
      # store only the last 7.955487 time units: one period of the in-phase wave at tau = 1
      .option(delay=4, meth="rk4", total=600, trans=592.044513, dt=0.01, maxstor=100000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")
m.auto.parameters("tau", "kappa", "om")


def exact_wave(tau, kappa, W):
    """Newton on Omega = om - kappa sin(Omega tau) from W; returns Omega, r"""
    for _ in range(30):
        W -= (W - OM + kappa * np.sin(W * tau)) / (1 + kappa * tau * np.cos(W * tau))
    return W, np.sqrt(1 - kappa + kappa * np.cos(W * tau))


# %% 2. the in-phase wave in tau, kappa = 1
m.run()
br = m.auto.run("tau", start="periodic", ds=0.01, dsmax=0.05, ntst=15, parmin=0.5, parmax=2.6, nmax=300)
trs = br.special("TR")
bt = np.array([p.x for p in br])
bT = np.array([p.period for p in br])
br_st = np.array([p.stable for p in br])
print(f"\n{len(br)} waves; torus points at tau = {[round(p.x, 5) for p in trs]}")

# %% 3. the rotating frame: z_j = exp(i W t) w_j, the wave is the equilibrium w1 = w2 = r.
# Its phase is free (a circle of equilibria); -1e-6 q1 pins it so that XPP's Newton step works
# and moves the zero root to about -1e-6.
c_, s_ = "kappa*cos(w*tau)", "kappa*sin(w*tau)"
rot = ModelBuilder("stuart_landau_pair_rotating").parameter(tau=1, kappa=1, om=OM, w=0.79)
for j, k in ((1, 2), (2, 1)):
    rot.ode(f"p{j}", f"p{j}-(om-w)*q{j}-(p{j}^2+q{j}^2)*p{j}+{c_}*delay(p{k},tau)+{s_}*delay(q{k},tau)-kappa*p{j}")
    rot.ode(f"q{j}", f"(om-w)*p{j}+q{j}-(p{j}^2+q{j}^2)*q{j}+{c_}*delay(q{k},tau)-{s_}*delay(p{k},tau)-kappa*q{j}"
            + ("-1e-6*q1" if j == 1 else ""))
rot.initial(p1=1, q1=0, p2=1, q2=0).option(delay=4)
mr = rot.load(HERE / "odes" / f"{NAME}_rotating.ode")


def rotating_roots(tau, kappa, period):
    W, r = exact_wave(tau, kappa, 2 * np.pi / period)
    mr.set(tau=tau, kappa=kappa, w=W, p1=r, q1=0, p2=r, q2=0)
    n0 = len(mr.log)
    mr.press("s", "g", True)
    rows = re.findall(r"^\s*\d+:\s+(\S+)\s+([+-])\s+i\s+(\S+)", mr.log[n0:], re.M)
    z = np.array([float(a) + 1j * (1 if s == "+" else -1) * float(b) for a, s, b in rows])
    cz = z[np.abs(z.imag) > 1e-3]
    return cz[np.argmin(np.abs(cz.real))]                # the complex pair nearest the axis


pairs = [rotating_roots(p.x, 1.0, p.period) for p in trs]
for p, z in zip(trs, pairs):
    print(f"  TR at tau = {p.x:.6f}: rotating-frame pair {z.real:+.1e} +- {abs(z.imag):.5f} i")

# %% 4. the torus curve in (tau, kappa), and a check of its end at kappa = 1.2
m.auto.axes(plot="two_par", param="tau", param2="kappa", xmin=1.3, xmax=2.5, ymin=0.7, ymax=1.2)
tcurve = m.auto.run(grab=trs[0].label, choice="two param", ds=0.02, dsmax=0.1, nmax=200)
end = tcurve[-1]
print(f"torus curve: {len(tcurve)} points from (tau, kappa) = (1.5840, 1) to ({end.x:.6f}, "
      f"{end.par['kappa']:.6f})")
curve_pairs = [rotating_roots(p.x, p.par["kappa"], p.period) for p in tcurve[::8]]
# a branch at kappa = 1.2: from the first wave (tau = 1, kappa = 1) continue in kappa up to
# 1.2 (Extend of a stored delay orbit in another parameter), then from there in tau
m.set(tau=1.0, kappa=1.0)
m.auto.axes(plot="hi", param="kappa")
k_br = m.auto.run(grab=br[0].label, choice="extend", ds=0.01, dsmax=0.05, parmin=0.5, parmax=1.2, nmax=100)
m.auto.axes(plot="hi", param="tau")
br12 = m.auto.run(grab=k_br[-1].label, choice="extend", ds=0.01, dsmax=0.05, parmin=0.5, parmax=2.0, nmax=300)
tr12 = br12.special("TR")
print(f"wave at kappa = {k_br[-1].x:.4f}: torus point at tau = {[round(p.x, 6) for p in tr12]}")
m.set(tau=1.0, kappa=1.0)

# %% 5. simulations on both sides of the first torus point (kappa = 1)
sims = {}
for tau in (1.4, 1.8):
    tr = m.run(keep=False, tau=tau, kappa=1.0, h1=1.0, x1=1.0, h2=0.9, x2=0.9, total=400, trans=0)
    sims[tau] = (tr.t, np.hypot(tr["x1"] - tr["x2"], tr["y1"] - tr["y2"]))

fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
tt = np.linspace(0.6, 2.6, 400)
W0 = np.array([exact_wave(t, 1.0, 2 * np.pi / np.interp(t, bt, bT))[0] for t in tt])
axs[0].plot(tt, 2 * np.pi / W0, color="0.8", lw=5, label="exact period")
axs[0].plot(bt[br_st], bT[br_st], "r.", ms=5, label="stable (Floquet)")
axs[0].plot(bt[~br_st], bT[~br_st], "k.", ms=4, label="unstable")
axs[0].plot([p.x for p in trs], [p.period for p in trs], "^", color="tab:blue", ms=9, label="torus (TR)")
axs[0].set(xlim=(0.9, 2.65), xlabel="tau", ylabel="period", title="In-phase wave, kappa = 1")
axs[0].legend(fontsize=8)
axs[1].plot([p.x for p in tcurve], [p.par["kappa"] for p in tcurve], "k-", lw=1.5, label="torus curve")
axs[1].plot([p.x for p in trs[:1]], [1.0], "^", color="tab:blue", ms=9, label="TR at kappa = 1")
axs[1].plot([p.x for p in tr12], [1.2] * len(tr12), "s", color="tab:orange", mfc="none", ms=10,
            label="TR of the branch at kappa = 1.2")
axs[1].set(xlim=(1.55, 1.68), ylim=(0.95, 1.22), xlabel="tau", ylabel="kappa",
           title="Torus curve in (tau, kappa)")
axs[1].legend(fontsize=8, loc="lower right")
for tau, col in ((1.4, "tab:red"), (1.8, "tab:blue")):
    axs[2].semilogy(*sims[tau], color=col, lw=1, label=f"tau = {tau}")
# (XPP stores solutions in single precision: below about 1e-7 there is only rounding noise)
axs[2].set(xlim=(0, 200), ylim=(1e-7, 5), xlabel="t", ylabel="|z1 - z2|",
           title="Simulations, kappa = 1: distance between z1 and z2")
axs[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "torus.png", dpi=130)

# %% checks
print("\nchecks:")
check("period of the in-phase wave at tau = 1", br[0].period, 7.955487, 1e-5)
err = [abs(2 * np.pi / p.period - exact_wave(p.x, 1.0, 2 * np.pi / p.period)[0]) for p in br]
check("largest |Omega - exact Omega| on the branch", max(err), 0, 1e-5)
# XPP's norm is the root mean square of (x1, y1, x2, y2) over the orbit: sqrt(2) r here
err = [abs(p.norm / np.sqrt(2) - exact_wave(p.x, 1.0, 2 * np.pi / p.period)[1]) for p in br]
check("largest |norm / sqrt(2) - r| on the branch", max(err), 0, 1e-4)
for p, ref, z in zip(trs, (1.584, 2.329), pairs):
    check(f"torus point near tau = {ref}", p.x, ref, 5e-4)
    check("  Re of the rotating-frame pair there", z.real, 0, 1e-4)
check("torus curve ends on the box edge kappa = 1.2", end.par["kappa"], 1.2, 1e-6)
check("rotating frame: largest |Re| of the pair along the torus curve",
      max(abs(z.real) for z in curve_pairs), 0, 1e-4)
check("torus point of the wave branch at kappa = 1.2 vs end of the torus curve",
      tr12[0].x if tr12 else np.nan, end.x, 1e-4)
d14, d18 = sims[1.4][1], sims[1.8][1]
check("tau = 1.4: perturbation of the in-phase wave decays (final / initial)", d14[-1] / d14[0], 0, 1e-3)
check("tau = 1.8: it does not (final distance > 0.1; 1 = yes)", float(d18[-1000:].max() > 0.1), 1, 0)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
