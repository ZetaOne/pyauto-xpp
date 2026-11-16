"""
A Stuart-Landau oscillator with delayed self-feedback: folds and tori of waves
=============================================================================

    z' = (mu + i om) z - |z|^2 z + kappa z(t-tau),     z = x + i y

A limit-cycle oscillator that feeds its own past state back.  Its periodic
solutions are rotating waves z = r exp(i Omega t) with

    Omega = om - kappa sin(Omega tau),     r^2 = mu + kappa cos(Omega tau),

so for larger tau there are several waves at once: the branch of waves
followed in tau folds back and forth.  The folds are exact:
1 + kappa tau cos(Omega tau) = 0.  For mu = 1, om = 10, kappa = 0.8 they are at
tau = 1.560254, 1.582413, 2.153950, 2.250648, 2.741942, 2.926149, ...

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. simulates one period of the stable wave at tau = 1 and continues it in
     tau with XPP 8.2's periodic-orbit solver for delay equations; it computes
     the Floquet multipliers, locates the folds (LP) and torus points (TR)
  3. checks the whole branch and the folds against the exact formulas
  4. checks the torus points independently: in a frame rotating with the wave,
     the wave is an equilibrium, and XPP's characteristic roots of that
     equilibrium (Sing pts) must have a pair on the imaginary axis there, with
     the same angle as the Floquet multipliers

Run:   python DELAYS_stuart-landau-single.py      (about 45 s: the continuation)
Figures go to figures/DELAYS_stuart-landau-single/.

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

try:
    HERE = Path(__file__).resolve().parent
    NAME = Path(__file__).stem
except NameError:            # run as a notebook in Jupyter: the kernel starts in this folder
    HERE = Path.cwd()
    NAME = "DELAYS_stuart-landau-single"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.7g} (expected {expected:.7g})")


MU, OM, KAP = 1.0, 10.0, 0.8

# %% 1. the model
# history (t < 0) = XPP's x(0)=, y(0)= formulas: the parameters x0, y0
mb = (ModelBuilder("stuart_landau_delay")
      .comment("Stuart-Landau oscillator with delayed self-feedback")
      .ode("x", "mu*x-om*y+kappa*delay(x,tau)-(x^2+y^2)*x")
      .ode("y", "om*x+mu*y+kappa*delay(y,tau)-(x^2+y^2)*y")
      .parameter(tau=1, mu=MU, om=OM, kappa=KAP, x0=1, y0=0)
      .initial_value("x", "x0")
      .initial_value("y", "y0")
      .initial(x=1, y=0)
      # store only the last 0.582759 time units: one period of the wave at tau = 1
      .option(delay=4, meth="rk4", total=300, trans=299.417241, dt=0.002, maxstor=200000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")
m.auto.parameters("tau", "mu", "om", "kappa")

# %% 2. one period of the wave, then the branch in tau
m.run()
waves = m.auto.run("tau", start="periodic", ds=0.01, dsmax=0.05, ntst=10, ncol=3,
                   parmin=0.5, parmax=3.0, nmax=600)
taus = np.array([p.x for p in waves])
Om = 2 * np.pi / np.array([p.period for p in waves])
# the wave is a circle of radius r: XPP's norm (root mean square of (x, y) over the mesh) is r,
# while max x over the mesh points misses the top of the circle slightly
rad = np.array([p.norm for p in waves])
stab = np.array([p.stable for p in waves])
lps, trs = waves.special("LP"), waves.special("TR")
print(f"\n{len(waves)} waves; folds at tau = {[round(p.x, 6) for p in lps]}")
print(f"torus points at tau = {[round(p.x, 5) for p in trs]}")

# %% 3. torus points, seen in the rotating frame
# z = exp(i W t) w turns the wave into the equilibrium w = r of
#   w' = (mu + i (om - W)) w - |w|^2 w + kappa exp(-i W tau) w(t - tau).
# That equilibrium lies on a circle of equilibria (any phase), which XPP's Newton step
# cannot handle; the term -1e-6 q pins the phase and moves the zero root to -1e-6 only.
rot = (ModelBuilder("stuart_landau_rotating")
       .ode("p", "mu*p-(om-w)*q-(p^2+q^2)*p+kappa*cos(w*tau)*delay(p,tau)+kappa*sin(w*tau)*delay(q,tau)")
       .ode("q", "(om-w)*p+mu*q-(p^2+q^2)*q+kappa*cos(w*tau)*delay(q,tau)-kappa*sin(w*tau)*delay(p,tau)"
            "-1e-6*q")
       .parameter(tau=1, mu=MU, om=OM, kappa=KAP, w=10.78)
       .initial(p=1, q=0)
       .option(delay=4))
mr = rot.load(HERE / "odes" / f"{NAME}_rotating.ode")


def rotating_roots(p):
    """XPP's characteristic roots (Sing pts, 'Print eigenvalues') at the wave of point p"""
    tau, W = p.x, 2 * np.pi / p.period
    for _ in range(20):                      # polish W on Omega = om - kappa sin(Omega tau)
        W -= (W - OM + KAP * np.sin(W * tau)) / (1 + KAP * tau * np.cos(W * tau))
    mr.set(tau=tau, w=W, p=np.sqrt(MU + KAP * np.cos(W * tau)), q=0)
    n0 = len(mr.log)
    mr.press("s", "g", True)
    rows = re.findall(r"^\s*\d+:\s+(\S+)\s+([+-])\s+i\s+(\S+)", mr.log[n0:], re.M)
    return np.array([float(a) + 1j * (1 if s == "+" else -1) * float(b) for a, s, b in rows])


tr_info = []
for p in trs:
    z = rotating_roots(p)
    pair = z[np.abs(z.imag) > 1e-3][0]                   # rightmost complex pair
    nu = abs(pair.imag)
    mu_c = [e for e in p.multipliers if abs(e.imag) > 1e-6]  # the complex Floquet multipliers
    th_floq = abs(np.angle(mu_c[0]))
    th_root = abs(np.angle(np.exp(1j * nu * p.period)))  # multiplier = exp(lambda T)
    tr_info.append((p, pair, th_floq, th_root, z))
    print(f"  TR at tau = {p.x:.5f}: rotating-frame pair {pair.real:+.1e} +- {nu:.5f} i; "
          f"angle of exp(i nu T) {th_root:.6f}, of the Floquet multipliers {th_floq:.6f}")

# %% figures
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
th = np.linspace(0, 40, 4000)                            # exact family: theta = Omega tau
W = OM - KAP * np.sin(th)
for ax, yy, yex, lab in ((axs[0], Om, W, "Omega = 2 pi / period"),
                         (axs[1], rad, np.sqrt(MU + KAP * np.cos(th)), "r (radius of the wave)")):
    ax.plot(th / W, yex, color="0.8", lw=5, label="exact waves")
    ax.plot(taus, np.where(stab, yy, np.nan), "r.", ms=4, label="stable (AUTO)")
    ax.plot(taus, np.where(stab, np.nan, yy), "k.", ms=3, label="unstable (AUTO)")
    sel = lambda ps: yy[[q.index - waves[0].index for q in ps]]           # noqa: E731
    ax.plot([p.x for p in lps], sel(lps), "s", color="tab:orange", ms=7, label="fold (LP)")
    ax.plot([p.x for p in trs], sel(trs), "^", color="tab:blue", ms=7, label="torus (TR)")
    ax.set(xlim=(0.9, 3.05), xlabel="tau", ylabel=lab)
axs[0].set_ylim(9.15, 10.85)
axs[1].set_ylim(0.3, 1.4)
fig.legend(*axs[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=5)
axs[0].set_title("Rotating waves in tau")
axs[1].set_title("Amplitude of the waves")
p, pair, _, _, z = tr_info[2]
axs[2].plot(z.real, z.imag, "o", mfc="none", ms=7)
axs[2].axvline(0, color="k", lw=0.8)
axs[2].set(xlim=(z.real.min() - 0.1, 0.15), xlabel="Re lambda", ylabel="Im lambda",
           title=f"Rotating frame, TR at tau = {p.x:.4f}")
axs[2].annotate(f"+-{abs(pair.imag):.4f} i", (0, abs(pair.imag)), xytext=(-60, 6),
                textcoords="offset points", fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(FIG / "waves.png", dpi=130)

# %% checks
print("\nchecks:")
check("largest |Omega - om + kappa sin(Omega tau)| on the branch",
      np.max(np.abs(Om - OM + KAP * np.sin(Om * taus))), 0, 1e-5)
check("largest |r^2 - mu - kappa cos(Omega tau)| on the branch",
      np.max(np.abs(rad**2 - MU - KAP * np.cos(Om * taus))), 0, 1e-4)
for ref in (1.560254, 1.582413, 2.153950, 2.250648, 2.741942, 2.926149):
    p = min(lps, key=lambda q: abs(q.x - ref))
    W = 2 * np.pi / p.period
    check(f"fold at tau = {ref}", p.x, ref, 2e-6)
    check("  1 + kappa tau cos(Omega tau) there", 1 + KAP * p.x * np.cos(W * p.x), 0, 1e-3)
check("number of torus points found", len(trs), 6, 0)
for p, pair, th_f, th_r, _ in tr_info:
    check(f"TR at tau = {p.x:.5f}: Re of the rotating-frame pair", pair.real, 0, 1e-4)
    check("  angle of exp(i nu T) vs Floquet multipliers", th_r, th_f, 1e-4)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
