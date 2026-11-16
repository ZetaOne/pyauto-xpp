"""
Mackey-Glass: a period-doubling cascade in a delay equation
===========================================================

    x' = beta x(t-tau) / (1 + x(t-tau)^10) - gamma x

A model of blood-cell production with a maturation delay tau.  With tau = 2,
gamma = 1 the equilibrium loses stability in a Hopf point at beta = 1.3369;
the periodic orbit born there period-doubles at beta = 1.5554491, the doubled
orbit again at 1.6958309, and again at 1.7199016, on the way to chaos.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. follows the equilibrium in beta with AUTO: two Hopf points, checked
     against the characteristic equation
  3. continues the periodic orbits from the first Hopf point; XPP 8.2 computes
     their Floquet multipliers and locates the period doubling (PD)
  4. switches branch at each PD ("doubling") to follow the orbits of period
     2, 4 and 8 times the first
  5. follows the first PD in two parameters (beta, tau)
  6. compares everything with an orbit diagram from simulations

Run:   python DELAYS_mackey-glass-period-doubling.py
Figures go to figures/DELAYS_mackey-glass-period-doubling/.

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
mb = (ModelBuilder("mackey_glass")
      .comment("Mackey-Glass equation")
      .ode("x", "beta*delay(x,tau)/(1+delay(x,tau)^10)-gamma*x")
      .parameter(beta=1.15, tau=2, gamma=1, x0=0.827197)
      .initial_value("x", "x0")    # history x(t) = x0 for t <= 0 (XPP's x(0)= formula)
      .initial(x=0.827197)         # x at t = 0: the equilibrium (beta-1)^(1/10) at beta = 1.15
      .option(delay=3, meth="rk4", total=100, dt=0.01, maxstor=100000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")
m.auto.parameters("beta", "tau", "gamma")          # tau must be in AUTO's list for step 5

# %% 2. equilibria in beta
m.run(total=100)
m.run_last(total=1)
eqs = m.auto.run("beta", start="steady", ds=0.01, dsmax=0.05, parmin=1, parmax=2.2, nmax=300)
hb = eqs.special("HB")
print("\nHopf points: beta =", [round(p.x, 6) for p in hb])

# %% 3.-4. periodic orbits and the period-doubling cascade
branches, pds = [], []
br = m.auto.run(grab=hb[0].label, choice="periodic", ds=0.01, dsmax=0.02, nmax=300, parmax=2.2)
branches.append(br)
for k in range(3):
    pd = br.special("PD")[0]
    pds.append(pd)
    print(f"  PD on the period-{2**k} branch at beta = {pd.x:.7f} (period {pd.period:.5f}, "
          f"multiplier {pd.multipliers[0].real:+.6f})")
    if k < 2:   # switch to the doubled branch and follow it
        br = m.auto.run(grab=pd.label, choice="doubling", ds=0.005, dsmax=0.01, nmax=300,
                        parmax=1.8)
        branches.append(br)
br8 = m.auto.run(grab=pds[2].label, choice="doubling", ds=0.002, dsmax=0.004, nmax=60, parmax=1.73)
branches.append(br8)
print(f"  period-8 orbits from beta = {br8[0].x:.5f} (period {br8[0].period:.4f})")

# %% 5. the first PD in the (beta, tau) plane
# the curve is followed inside the Two par window; ending it at beta = 2 gives tau there
m.auto.axes(plot="two_par", param="beta", param2="tau", xmin=1.3, xmax=2.0, ymin=1.0, ymax=2.3)
pd_up = m.auto.run(grab=pds[0].label, choice="two param", ds=0.01, dsmax=0.05, nmax=200)
pd_dn = m.auto.run(grab=pds[0].label, choice="two param", ds=-0.01, dsmax=0.05, nmax=200)
hb_curve = [m.auto.run(grab=hb[0].label, choice="two param", ds=s * 0.01, dsmax=0.05, nmax=120,
                       parmax=2.0) for s in (1, -1)]
tau_at_2 = pd_up[-1].par["tau"]
print("  parameters after the two-parameter runs:", {k: round(m.get(k), 4) for k in ("beta", "tau")})
m.set(tau=2.0)                                   # AUTO leaves the last point's parameters behind
print(f"  PD curve: from (beta, tau) = ({pd_dn[-1].x:.4f}, {pd_dn[-1].par['tau']:.4f}) "
      f"to ({pd_up[-1].x:.4f}, {tau_at_2:.6f})")

fig, ax = plt.subplots(figsize=(5.5, 4.2))
for c, col, lab in ((pd_up, "tab:blue", "period doubling"), (pd_dn, "tab:blue", None),
                    (hb_curve[0], "tab:purple", "Hopf of the equilibrium"), (hb_curve[1], "tab:purple", None)):
    ax.plot([p.x for p in c], [p.par["tau"] for p in c], "-", color=col, lw=2, label=lab)
ax.plot(pds[0].x, 2, "ko", ms=6)
ax.plot(2.0, tau_at_2, "ks", ms=6)
ax.annotate("first PD\n(tau = 2)", (pds[0].x, 2), xytext=(8, -4), textcoords="offset points", fontsize=8)
ax.annotate(f"tau = {tau_at_2:.4f}", (2.0, tau_at_2), xytext=(-45, -18), textcoords="offset points",
            fontsize=8)
ax.set(xlim=(1.0, 2.05), ylim=(1.0, 2.3), xlabel="beta", ylabel="tau", title="Two-parameter curves")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "two_parameter.png", dpi=130)

# %% 6. orbit diagram from simulations, with the AUTO branches on top
betas = np.linspace(1.3, 2.1, 161)
fig, ax = plt.subplots(figsize=(8, 4.8))
for b in betas:
    tr = m.run(keep=False, beta=b, x=0.9, x0=0.9, total=500, trans=300)
    x = tr["x"]
    pk = x[1:-1][(x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])]          # local maxima
    ax.plot(np.full(len(pk), b), pk, "k.", ms=1.2, zorder=3)
ax.plot([p.x for p in eqs], [p.u0["x"] for p in eqs], "k-", lw=0.8)
for k, (c, col) in enumerate(zip(branches, ["tab:red", "tab:orange", "tab:green", "tab:blue"])):
    xs = np.array([p.x for p in c])
    ys = np.array([p.umax["x"] for p in c])
    st = np.array([p.stable for p in c])
    ax.plot(xs, np.where(st, ys, np.nan), "-", color=col, lw=3, alpha=0.6, label=f"period-{2**k} orbit, stable")
    ax.plot(xs, np.where(st, np.nan, ys), ":", color=col, lw=1.5)
ax.plot([], [], "k:", lw=1.5, label="unstable orbits")
ax.plot([p.x for p in pds], [p.umax["x"] for p in pds], "kD", ms=5, label="PD (AUTO)")
ax.plot([p.x for p in hb], [p.u0["x"] for p in hb], "o", color="purple", ms=7, label="Hopf")
ax.set(xlim=(1.3, 2.1), ylim=(0.7, 1.45), xlabel="beta", ylabel="x",
       title="Local maxima of x in simulations (dots) and max x on AUTO's orbits")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "orbit_diagram.png", dpi=130)


def sim_period(b, level=1.0):
    tr = m.run(keep=False, beta=b, x=0.9, x0=0.9, total=700, trans=500)
    t, x = tr.t, tr["x"]
    up = np.where((x[:-1] < level) & (x[1:] >= level))[0]
    tc = t[up] + (level - x[up]) / (x[up + 1] - x[up]) * (t[up + 1] - t[up])
    return tc, x


# %% 7. checks
print("\nchecks:")
# Hopf: lambda = i w in  lambda = -1 + c exp(-lambda tau),  c = f'(x*) = (10 - 9 beta)/beta,
# x* = (beta-1)^(1/10).  Then c cos(w tau) = 1, -c sin(w tau) = w: tan(2w) = -w with
# 2w in (pi/2, pi) + 2 pi k, c = -sqrt(1 + w^2), beta = 10/(9 + c).
for k, p in enumerate(hb):
    lo, hi = (np.pi / 2 + 2 * k * np.pi) / 2 + 1e-9, (np.pi + 2 * k * np.pi) / 2 - 1e-9
    for _ in range(80):                                  # bisection on tan(2w) + w
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.tan(2 * mid) + mid < 0 else (lo, mid)
    w = (lo + hi) / 2
    check(f"Hopf point {k + 1} (characteristic equation)", p.x, 10 / (9 - np.sqrt(1 + w * w)), 2e-5)
    check(f"period at Hopf point {k + 1} = 2 pi / w", p.period, 2 * np.pi / w, 1e-3)
for pd, ref in zip(pds, (1.5554491, 1.6958309, 1.7199016)):
    check(f"PD at beta = {ref}", pd.x, ref, 1e-6)
    check("  Floquet multiplier there = -1", pd.multipliers[0].real, -1, 1e-4)
for k in range(1, 4):
    check(f"first period-{2**k} orbit has twice the period at the PD", branches[k][0].period,
          2 * pds[k - 1].period, 0.01 * pds[k - 1].period)
check("PD curve passes beta = 2 at tau", tau_at_2, 1.324922, 1e-5)
# simulations: x crosses 1 upward once per loop; the period-2 orbit at beta = 1.65 makes two loops
for b, k in ((1.5, 0), (1.65, 1)):
    tc, x = sim_period(b)
    c = branches[k]
    T_auto = np.interp(b, [p.x for p in c], [p.period for p in c])
    check(f"period at beta = {b}: simulation vs AUTO period-{2**k} branch",
          np.mean(tc[2**k:] - tc[:-2**k]), T_auto, 2e-3)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
