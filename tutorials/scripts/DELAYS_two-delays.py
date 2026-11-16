"""
A linear equation with two delays: characteristic roots and Hopf points
=======================================================================

    x' = -a ( x(t-1) + x(t-1.7) )

The simplest test of how XPP handles several delays at once.  The
characteristic equation  lambda + a (exp(-lambda) + exp(-1.7 lambda)) = 0  has
infinitely many roots.  For small a they all have negative real part; at
a = 0.633594 a pair crosses the imaginary axis at +-1.163553 i, and at
a = 4.251022 another pair crosses at +-8.14487 i.  The equation is linear, so
there are no periodic orbits to continue: past the first crossing solutions
oscillate with growing amplitude.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. follows the equilibrium x = 0 in a with AUTO (which uses the delay
     equation's characteristic roots for stability) and finds both Hopf points
  3. derives the exact Hopf points and checks them
  4. prints XPP's characteristic roots at a = 0.8 (Sing pts with "Print
     eigenvalues") and checks them against the characteristic equation
  5. simulates on both sides of the first Hopf point: the growth or decay rate
     and the frequency are those of the rightmost root

Run:   python DELAYS_two-delays.py
Figures go to figures/DELAYS_two-delays/.

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
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.8g} (expected {expected:.8g})")


# %% 1. the model
# The history (x for t < 0) is XPP's x(0)= formula, here the parameter x0; a run from a
# constant history v sets both x0=v and x=v.
mb = (ModelBuilder("two_delays")
      .comment("linear equation with two delays")
      .ode("x", "-a*(delay(x,1)+delay(x,1.7))")
      .parameter(a=0.2, x0=0.0)
      .initial_value("x", "x0")
      .initial(x=0.0)
      .option(delay=2, meth="rk4", total=100, dt=0.01, bound=1000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")


def char(lam, a):
    return lam + a * (np.exp(-lam) + np.exp(-1.7 * lam))


def delay_roots(a):
    """XPP's characteristic roots of the equilibrium x = 0: Sing pts > Go, answering yes to
    "Print eigenvalues?"; XPP prints the rightmost roots to its log"""
    m.set(a=a, x=0.0)
    n0 = len(m.log)
    m.press("s", "g", True)
    rows = re.findall(r"^\s*\d+:\s+(\S+)\s+([+-])\s+i\s+(\S+)", m.log[n0:], re.M)
    return np.array([float(r) + 1j * (1 if s == "+" else -1) * float(i) for r, s, i in rows])


# %% 2. AUTO: the equilibrium x = 0 in a
m.run(total=1)
m.run_last(total=1)
# small steps: AUTO locates a Hopf point of a delay equation to within about dsmax/10
eqs = m.auto.run("a", start="steady", ds=0.01, dsmax=0.002, parmin=0, parmax=5, nmax=5000)
hbs = eqs.special("HB")
for p in hbs:
    print(f"\nHopf at a = {p.x:.6f}, omega = 2 pi/period = {2 * np.pi / p.period:.6f}")

# %% 3. characteristic roots at a = 0.5 and a = 0.8; the rightmost root along a
roots = {a: delay_roots(a) for a in (0.5, 0.8)}
print("\ncharacteristic roots at a = 0.8 (XPP, Sing pts):")
for z in roots[0.8][:6]:
    print(f"   {z.real:+.8f} {z.imag:+.8f} i    |residual| {abs(char(z, 0.8)):.1e}")
a_grid = np.linspace(0.3, 5, 95)
top = [delay_roots(a) for a in a_grid]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
for a, mk in ((0.5, "o"), (0.8, "s")):
    a1.plot(roots[a].real, roots[a].imag, mk, mfc="none", ms=7, label=f"a = {a}")
a1.axvline(0, color="k", lw=0.8)
a1.set(xlabel="Re lambda", ylabel="Im lambda", title="Rightmost characteristic roots (XPP)")
a1.legend(fontsize=8)
for k in range(4):                                         # the 4 rightmost pairs
    a2.plot(a_grid, [np.sort(np.unique(np.round(r.real, 9)))[::-1][k] for r in top], "-", lw=1.5,
            label="real parts, 4 rightmost pairs" if k == 0 else None, color=f"C{k}")
a2.axhline(0, color="k", lw=0.8)
a2.plot([p.x for p in hbs], [0] * len(hbs), "o", color="purple", ms=8, label="Hopf (AUTO)")
a2.set(ylim=(-2.5, 1.2), xlabel="a", ylabel="Re lambda", title="Crossings of the imaginary axis")
a2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "roots.png", dpi=130)

# %% 4. simulations either side of the first Hopf point
sims = {}
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, (a, col) in zip(axs, ((0.5, "tab:blue"), (0.8, "tab:red"))):
    tr = m.run(keep=False, a=a, x=0.1, x0=0.1, total=60)
    t, x = tr.t, tr["x"]
    pk = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 0))[0] + 1
    pk = pk[t[pk] > 15]                                    # after the faster roots have died out
    rate = np.polyfit(t[pk], np.log(x[pk]), 1)[0]          # growth rate of the peaks
    freq = 2 * np.pi / np.diff(t[pk]).mean()
    sims[a] = (rate, freq)
    lam = roots[a][0]
    env = x[pk[0]] * np.exp(lam.real * (t - t[pk[0]]))
    ax.plot(t, x, color=col, lw=1, label="simulation")
    ax.plot(t[pk], x[pk], "k.", ms=5, label="peaks used for the rate")
    ax.plot(t, env, "k--", lw=0.8, label="+-exp(Re lambda t), rightmost root")
    ax.plot(t, -env, "k--", lw=0.8)
    ax.set(xlabel="t", ylabel="x", title=f"a = {a}: rightmost root {lam.real:+.4f} +- {lam.imag:.4f} i")
    ax.legend(fontsize=7, loc="upper right" if a < 0.6 else "upper left")
    if a < 0.6:
        ax.set_ylim(-0.12, 0.12)
    print(f"a = {a}: peaks grow at rate {rate:+.5f} with frequency {freq:.5f}; "
          f"rightmost root {lam.real:+.5f} +- {lam.imag:.5f} i")
fig.tight_layout()
fig.savefig(FIG / "simulations.png", dpi=130)

# %% 5. checks
print("\nchecks:")
# lambda = i w:  cos w + cos 1.7w = 2 cos(1.35 w) cos(0.35 w) = 0  and  w = a (sin w + sin 1.7w).
# Candidates w = (2k+1) pi/2.7 and (2k+1) pi/0.7; the Hopf points are those with 0 < a < 5.
ws = np.sort(np.r_[(2 * np.arange(10) + 1) * np.pi / 2.7, (2 * np.arange(4) + 1) * np.pi / 0.7])
aa = ws / (np.sin(ws) + np.sin(1.7 * ws))
keep = (aa > 0) & (aa < 5)
check("number of Hopf points in 0 < a < 5", len(hbs), keep.sum(), 0)
for p, a_h, w_h in zip(hbs, aa[keep], ws[keep]):
    check(f"Hopf at a = {a_h:.6f} (exact)", p.x, a_h, 2e-6)
    check(f"  omega there = {w_h:.6f}", 2 * np.pi / p.period, w_h, 2e-5)
r8 = roots[0.8]
for ref in (0.11191770 + 1.24352376j, -1.33013388 + 4.45755707j, -1.17819739 + 8.12643108j):
    check(f"root at a = 0.8 near {ref:.4f}: distance", np.min(np.abs(r8 - ref)), 0, 1e-7)
check("the rightmost root at a = 0.8", r8[0].real, 0.11191770, 1e-8)
# XPP prints 10 significant digits, so the residual of a printed root is about 1e-9 |lambda|
check("largest |characteristic function| / |lambda| at the 12 roots, a = 0.8",
      max(abs(char(z, 0.8)) / abs(z) for z in r8), 0, 1e-8)
m.set(a=0.8)
check("m.equilibrium's eigenvalue at a = 0.8 is the rightmost root",
      abs(m.equilibrium({"x": 0}).eigenvalues[0] - r8[0]), 0, 1e-7)
for a in (0.5, 0.8):
    check(f"a = {a}: growth rate of simulated peaks vs Re of rightmost root",
          sims[a][0], roots[a][0].real, 2e-3)
    check(f"a = {a}: simulated frequency vs Im of rightmost root", sims[a][1], roots[a][0].imag, 2e-3)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
