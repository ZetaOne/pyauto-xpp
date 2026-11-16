"""
Interacting saddle-nodes
========================

From Ashwin, Postlethwaite and Sieber (Nonlinearity 2026,
DOI 10.1088/1361-6544/ae788a):
    x' = x^2 - lambda + 2 alpha y - gamma,   y' = y^2 - lambda + 2 beta x + gamma,
with lambda = mu + gamma (gamma + alpha^2 - beta^2)/(alpha + beta)^2.  Both
nullclines are parabolas, so there are up to four equilibria, and they appear
and disappear in pairs at saddle-node bifurcations.  The shift built into
lambda puts a Hopf point exactly at mu = 0, between two saddle-nodes close by.

The model is integrated with Dormand-Prince 5 (XPP's "5dp") and bound = 10.
(XPP reads only the first letter of meth, so "dopri5" or "dormprince" would
make it a discrete map.)

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. draws the phase plane at mu = -0.005: the two parabolas, the four
     equilibria and trajectories (a zoom shows the three close together)
  3. counts the equilibria against mu from the exact quartic, and follows
     them in mu with AUTO: two folds (LP) and the Hopf point
  4. checks the equilibria against the polynomial equations, and the folds
     and the Hopf point against exact values

Run:   python 18_2D-FLOWS_interacting-saddle-nodes.py
Figures go to figures/18_2D-FLOWS_interacting-saddle-nodes/.

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
    NAME = "18_2D-FLOWS_interacting-saddle-nodes"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
g, a, b = 3.58, 3.1, -1.3                            # gamma, alpha, beta
shift = g * (g + a**2 - b**2) / (a + b)**2           # lambda = mu + shift
mb = (ModelBuilder("interacting_saddle_nodes")
      .comment("Interacting saddle-nodes (Ashwin, Postlethwaite and Sieber, Nonlinearity 2026)")
      .parameter(mu=-0.005, gamma=g)
      .parameter(alpha=a, beta=b)
      .fixed("lambda", "mu + gamma*(gamma+alpha^2-beta^2)/(alpha+beta)^2")
      .ode("x", "x^2-lambda+2*alpha*y-gamma")
      .ode("y", "y^2-lambda+2*beta*x+gamma")
      .initial(x=1, y=1)
      .option(meth="5dp", total=10, dt=0.01, toler=1e-6, bound=10))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
m.auto.parameters("mu", "gamma", "alpha", "beta")
m.buttons.numerics.sing_pt_ctrl(500, 1e-10, 1e-8)   # accurate Newton and eigenvalues


def exact_equilibria(mu):
    """real roots of the quartic: y = (s - x^2)/(2 alpha) with s = lambda + gamma, put into y' = 0"""
    lam = mu + shift
    s = lam + g
    r = np.roots([1, 0, -2 * s, 8 * a**2 * b, s**2 + 4 * a**2 * (g - lam)])
    x = np.sort(r[abs(r.imag) < 1e-9].real)
    return x, (s - x**2) / (2 * a)


# %% 2. phase plane at mu = -0.005
ex, ey = exact_equilibria(-0.005)
eqs = [m.equilibrium({"x": x, "y": y}) for x, y in zip(ex, ey)]   # XPP's Newton from each root
kind = lambda e: ("saddle" if np.prod(e.eigenvalues.real) < 0 else
                  ("stable " if e.stable else "unstable ") + ("focus" if abs(e.eigenvalues.imag).max() > 0 else "node"))
print("equilibria at mu = -0.005:")
for e in eqs:
    print(f"  x = {e.x['X']:+.6f}, y = {e.x['Y']:+.6f}  {kind(e)}, eigenvalues {np.round(e.eigenvalues, 4)}")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, win in zip(axs, ((-8.5, 19.6, -13.2, 5.9), (-3.0, -1.6, 1.2, 2.3))):
    nc = m.nullclines(x="x", y="y", xlo=win[0], xhi=win[1], ylo=win[2], yhi=win[3], mesh=300)
    ax.plot(nc["X"][:, 0], nc["X"][:, 1], color="tab:orange", lw=2, label="x' = 0")
    ax.plot(nc["Y"][:, 0], nc["Y"][:, 1], color="tab:green", lw=2, label="y' = 0")
    ax.set(xlim=win[:2], ylim=win[2:], xlabel="x", ylabel="y")
starts = [((1, 1), axs[0]), ((-6, -4), axs[0]), ((4, -8), axs[0]),
          ((-2.0, 2.05), axs[1]), ((-2.3, 1.2), axs[1]), ((-2.9, 2.2), axs[1]), ((-2.35, 1.75), axs[1])]
for ((x0, y0), ax), col in zip(starts, ["tab:blue", "tab:purple", "tab:red", "tab:blue", "tab:purple",
                                        "tab:red", "tab:brown"]):
    tr = m.run(keep=False, x=x0, y=y0, total=10)     # stops early if |x| or |y| passes 10 (bound)
    ax.plot(tr["x"], tr["y"], color=col, lw=1.2)
    ax.plot(x0, y0, "k.", ms=6)
for ax in axs:
    for e in eqs:
        ax.plot(e.x["X"], e.x["Y"], "o", mfc="k" if e.stable else "white", mec="k", ms=7)
axs[0].add_patch(plt.Rectangle((-3.0, 1.2), 1.4, 1.1, fill=False, ls="--", color="0.4"))
axs[0].set_title("mu = -0.005: two parabolas, four equilibria")
axs[0].legend(fontsize=8, loc="lower left")
axs[1].set_title("zoom: node, saddle, unstable focus")
for e in eqs[:3]:
    axs[1].annotate(kind(e), (e.x["X"], e.x["Y"]), xytext=(6, -12), textcoords="offset points", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "phase_plane.png", dpi=130)

# %% 3. equilibria against mu: exact count, and AUTO from the saddle
e0 = eqs[1]                                           # the saddle at mu = -0.005
parts = []
for sign in (1, -1):
    m.auto.reset()
    m.set(mu=-0.005, x=e0.x["X"], y=e0.x["Y"])
    m.run(total=1e-4)                                 # make the saddle the current state
    parts.append(m.auto.run("mu", start="steady", ds=sign * 0.0005, dsmax=0.002,
                            parmin=-0.05, parmax=0.05, nmax=2000))
lps = sorted((p for d in parts for p in d.special("LP")), key=lambda p: p.x)
hb = [p for d in parts for p in d.special("HB")][0]
print(f"\nAUTO: folds at mu = {lps[0].x:.7f} and {lps[1].x:.7f}; Hopf at mu = {hb.x:.2e}")

mus = np.linspace(-0.05, 0.05, 2001)
count = np.array([len(exact_equilibria(u)[0]) for u in mus])
fig, axs = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
for d in parts:
    xs = np.array([p.x for p in d])
    us = np.array([p.u0["x"] for p in d])
    st = np.array([bool(p.stable) for p in d])
    axs[0].plot(xs, np.where(st, us, np.nan), "r-", lw=2.5)
    axs[0].plot(xs, np.where(~st, us, np.nan), "k-", lw=1)
axs[0].plot([], [], "r-", lw=2.5, label="stable (AUTO)")
axs[0].plot([], [], "k-", label="unstable (AUTO)")
axs[0].plot([p.x for p in lps], [p.u0["x"] for p in lps], "s", color="purple", ms=7, label="fold (LP)")
axs[0].plot(hb.x, hb.u0["x"], "o", color="tab:blue", ms=7, label="Hopf")
sel = mus[::50]
for u in sel:
    xr = exact_equilibria(u)[0]
    axs[0].plot([u] * len(xr[xr < 0]), xr[xr < 0], ".", color="0.5", ms=3)
axs[0].plot([], [], ".", color="0.5", label="exact roots")
axs[0].set(ylabel="x", ylim=(-3.0, -1.75), title="Equilibria near x = -2.3 against mu")
axs[0].legend(fontsize=8, loc="upper right")
axs[1].step(mus, count, where="mid", color="tab:green")
axs[1].set(xlabel="mu", ylabel="number", xlim=(-0.052, 0.052), ylim=(1, 5), yticks=[2, 4])
fig.tight_layout()
fig.savefig(FIG / "equilibria_mu.png", dpi=130)

# %% 4. checks
print("\nchecks:")
res = max(max(abs(e.x["X"]**2 - (-0.005 + shift) + 2 * a * e.x["Y"] - g),
              abs(e.x["Y"]**2 - (-0.005 + shift) + 2 * b * e.x["X"] + g)) for e in eqs)
check("XPP equilibria: largest residual of the two equations", res, 0, 1e-10)
check("XPP equilibria vs quartic roots (largest |dx|)", max(abs(e.x["X"] - x) for e, x in zip(eqs, ex)), 0, 1e-10)
check("number of equilibria at mu = -0.005", len(eqs), 4, 0)
# exact folds: a double root of the quartic Q(x) needs Q = Q' = 0; Q' = 0 gives
# s = x^2 + 2 alpha^2 beta / x, and then Q = 0 becomes
# -x^4 + 2 beta x^3 + 2 gamma x^2 - 2 alpha^2 beta x + alpha^2 beta^2 = 0
xr = np.roots([-1, 2 * b, 2 * g, -2 * a**2 * b, a**2 * b**2])
mu_sn = np.sort([x**2 + 2 * a**2 * b / x - g - shift for x in xr.real[abs(xr.imag) < 1e-12]])
mu_sn = mu_sn[abs(mu_sn) < 0.05]
check("fold 1 in mu (exact)", lps[0].x, mu_sn[0], 1e-6)
check("fold 2 in mu (exact)", lps[1].x, mu_sn[1], 1e-6)
check("count jumps 2 -> 4 -> 2 at the folds (sum of |jumps| / 2)", np.abs(np.diff(count)).sum() / 2, 2, 0)
# exact Hopf: on y = -x the trace 2x + 2y vanishes; the equations then give x = -gamma/(alpha+beta)
# and lambda = shift, so mu = 0; det = 4(xy - alpha beta) gives omega = 2 sqrt(-alpha beta - x^2)
xh = -g / (a + b)
check("Hopf in mu (exact 0)", hb.x, 0, 1e-5)
check("Hopf x = -gamma/(alpha+beta) (exact)", hb.u0["x"], xh, 1e-4)
check("Hopf frequency (exact)", m.auto.lyapunov(hb)[1], 2 * np.sqrt(-a * b - xh**2), 1e-4)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
