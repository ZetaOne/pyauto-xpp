"""
Wilson-Cowan: excitatory and inhibitory populations
===================================================

Mean activities u (excitatory) and v (inhibitory) of two coupled neural
populations with a sigmoid response f:
    taue u' = -u + f(aee u - aei v - te),   taui v' = -v + f(aie u - aii v - ti).
Along the excitatory threshold te the equilibria form an S-shaped branch with
two folds; and when inhibition is slow enough (large taui) the rest state
loses stability and the network oscillates.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. draws the phase plane at te = 2, taui = 6: nullclines, the unstable
     equilibrium and the limit cycle around it
  3. follows the equilibria in te with AUTO (small steps: the folds are sharp)
     and finds the two folds and a Hopf point
  4. follows the equilibrium at te = 2 in taui: the Hopf point where slow
     inhibition starts the oscillation, and the orbits born there; the same
     amplitude and period from plain simulations
  5. checks the folds, the Hopf points and the period against exact values

Run:   python 2D-FLOWS_wilson-cowan.py
Figures go to figures/2D-FLOWS_wilson-cowan/.

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
    NAME = "2D-FLOWS_wilson-cowan"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
P = dict(aee=13, aei=10, te=2, taue=2, aie=16, aii=4, ti=5, taui=6)
mb = (ModelBuilder("wilson_cowan")
      .comment("Wilson-Cowan model")
      .function("f(u)", "1/(1+exp(-u))")
      .ode("u", "(-u+f(aee*u-aei*v-te))/taue")
      .ode("v", "(-v+f(aie*u-aii*v-ti))/taui")
      .parameter(**P)
      .initial(u=0.1, v=0.1)
      .option(total=200, dt=0.05))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
m.auto.parameters("te", "taui", "ti", "aee", "aei", "aie", "aii", "taue")
m.buttons.numerics.sing_pt_ctrl(500, 1e-10, 1e-8)   # accurate Newton and eigenvalues

# %% 2. phase plane at te = 2, taui = 6
win = (-0.05, 1.05, -0.05, 1.0)
nc = m.nullclines(x="u", y="v", xlo=win[0], xhi=win[1], ylo=win[2], yhi=win[3], mesh=300)
tr = m.run(keep=False, u=0.1, v=0.1, total=300)
cyc = m.run(keep=False, u=0.1, v=0.1, total=400, trans=300)
eq = m.equilibrium({"u": 0.34, "v": 0.31})
print("equilibrium at te = 2:", eq, " eigenvalues", np.round(eq.eigenvalues, 4))

fig, ax = plt.subplots(figsize=(6, 5.2))
ax.plot(nc["U"][:, 0], nc["U"][:, 1], color="tab:orange", lw=2, label="u' = 0")
ax.plot(nc["V"][:, 0], nc["V"][:, 1], color="tab:green", lw=2, label="v' = 0")
ax.plot(tr["u"], tr["v"], color="tab:blue", lw=0.8, alpha=0.6, label="from (0.1, 0.1)")
ax.plot(cyc["u"], cyc["v"], color="tab:red", lw=2, label="limit cycle")
ax.plot(eq.x["U"], eq.x["V"], "o", mfc="white", mec="k", ms=9, label="unstable equilibrium")
ax.set(xlim=win[:2], ylim=win[2:], xlabel="u (excitatory)", ylabel="v (inhibitory)",
       title="Wilson-Cowan, te = 2, taui = 6")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "phase_plane.png", dpi=130)

# %% 3. AUTO: equilibria in te (taui = 6)
m.set(te=-3)
m.run(total=300)                 # rest on the high-activity state at te = -3
m.run_last(total=10)             # AUTO starts from the current state
eqt = m.auto.run("te", start="steady", ds=0.002, dsmax=0.01, parmin=-4, parmax=6, nmax=4000)
lp_te = sorted(p.x for p in eqt.special("LP"))
hb_te = [p.x for p in eqt.special("HB")]
print("\nAUTO in te: folds at", np.round(lp_te, 5), " Hopf at", np.round(hb_te, 5))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
xs = np.array([p.x for p in eqt])
us = np.array([p.u0["u"] for p in eqt])
st = np.array([bool(p.stable) for p in eqt])
ax.plot(xs, np.where(st, us, np.nan), "r-", lw=2.5, label="stable")
ax.plot(xs, np.where(~st, us, np.nan), "k-", lw=1, label="unstable")
for kind, mk in (("LP", "s"), ("HB", "o")):
    pts = eqt.special(kind)
    ax.plot([p.x for p in pts], [p.u0["u"] for p in pts], mk, color="purple", ms=7, label=kind)
ax.set(xlim=(-3.2, 6), xlabel="te", ylabel="u", title="Equilibria in te (taui = 6)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "equilibria_te.png", dpi=130)

# %% 4. slow inhibition: equilibria and orbits in taui at te = 2
m.auto.reset()
m.set(te=2, taui=1)
m.run(total=300)
m.run_last(total=10)
eqi = m.auto.run("taui", start="steady", ds=0.002, dsmax=0.01, parmin=0.5, parmax=15, nmax=4000)
hb = eqi.special("HB")[0]
l1, om = m.auto.lyapunov(hb)
print(f"\nAUTO in taui: Hopf at taui = {hb.x:.5f}, l1 = {l1:.3f}, period at onset {2 * np.pi / om:.4f}")
po = m.auto.run(grab=hb.label, choice="periodic", ntst=60, ds=0.002, dsmax=0.05, nmax=500, parmax=15)

T = np.linspace(1.0, 14.0, 27)                       # the same from simulations
amp, per = [], []
for ti_ in T:
    s = m.run(keep=False, te=2, taui=ti_, u=0.1, v=0.1, total=1500, trans=1000)
    up = np.where((s["u"][:-1] < 0.5) & (s["u"][1:] >= 0.5))[0]
    amp.append(np.ptp(s["u"]))
    per.append(np.diff(s.t[up]).mean() if len(up) > 2 else np.nan)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
px = np.array([p.x for p in po])
axs[0].plot(px, [p.umax["u"] - p.umin["u"] for p in po], "-", color="tab:green", lw=2, label="AUTO orbits")
axs[0].plot(T, amp, "o", color="tab:blue", ms=4, label="simulation")
axs[1].plot(px, [p.period for p in po], "-", color="tab:green", lw=2, label="AUTO orbits")
axs[1].plot(T, per, "o", color="tab:blue", ms=4, label="simulation")
for ax, lab in zip(axs, ("amplitude of u (max - min)", "period")):
    ax.axvline(hb.x, color="purple", ls=":", label=f"Hopf, taui = {hb.x:.3f}")
    ax.set(xlabel="taui", ylabel=lab, xlim=(0.8, 14.5))
    ax.legend(fontsize=8)
fig.suptitle("Slow inhibition makes the network oscillate (te = 2)")
fig.tight_layout()
fig.savefig(FIG / "oscillations_taui.png", dpi=130)


# %% 5. checks
def root(fun, a, b, n=200):
    """bisection for a sign change of fun in [a, b]"""
    for _ in range(n):
        c = 0.5 * (a + b)
        a, b = (c, b) if np.sign(fun(c)) == np.sign(fun(a)) else (a, c)
    return 0.5 * (a + b)


def roots(fun, lo, hi):
    """all sign changes of fun on a fine grid, refined by bisection"""
    x = np.linspace(lo, hi, 20001)
    y = fun(x)
    return [root(fun, x[k], x[k + 1]) for k in np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]]


# exact: on an equilibrium, aie u - aii v - ti = logit(v), so u is explicit in v, and then
# te = aee u - aei v - logit(u): the whole branch is a curve te(v), and f' = u(1-u), v(1-v) there
logit = lambda x: np.log(x / (1 - x))
u_of = lambda v: (logit(v) + P["aii"] * v + P["ti"]) / P["aie"]
te_of = lambda v: P["aee"] * u_of(v) - P["aei"] * v - logit(u_of(v))
dte = lambda v: ((P["aee"] - 1 / (u_of(v) * (1 - u_of(v)))) * (1 / (v * (1 - v)) + P["aii"]) / P["aie"]
                 - P["aei"])                       # d te / d v: zero at a fold
# u_of increases with v; keep 0 < u < 1
vgrid = (root(u_of, 1e-12, 0.5) + 1e-9, root(lambda v: u_of(v) - 1, 0.5, 1 - 1e-12) - 1e-9)
te_folds = sorted(te_of(np.array(roots(dte, *vgrid))))
print("\nchecks:")
check("fold 1 in te (exact)", lp_te[0], te_folds[0], 1e-4)
check("fold 2 in te (exact)", lp_te[1], te_folds[1], 1e-4)


def trace(v, taui):           # trace of the Jacobian on the branch
    u = u_of(v)
    return (-1 + P["aee"] * u * (1 - u)) / P["taue"] - (1 + P["aii"] * v * (1 - v)) / taui


vh = roots(lambda v: trace(v, 6), *vgrid)
check("Hopf in te at taui = 6 (exact, trace = 0)", hb_te[0], te_of(vh[0]), 1e-4)
# at te = 2: the equilibrium solves te(v) = 2; it does not depend on taui, and the trace
# vanishes at taui = taue (1 + aii v(1-v)) / (aee u(1-u) - 1)
v2 = roots(lambda v: te_of(v) - 2, *vgrid)[0]
u2 = u_of(v2)
fu, fv = u2 * (1 - u2), v2 * (1 - v2)
taui_h = P["taue"] * (1 + P["aii"] * fv) / (P["aee"] * fu - 1)
check("equilibrium u at te = 2 (exact)", eq.x["U"], u2, 1e-6)
check("Hopf in taui (exact)", hb.x, taui_h, 1e-4)
det = (-(-1 + P["aee"] * fu) * (1 + P["aii"] * fv) + P["aei"] * fu * P["aie"] * fv) / (P["taue"] * taui_h)
check("period at onset 2*pi/sqrt(det) (exact)", 2 * np.pi / om, 2 * np.pi / np.sqrt(det), 1e-3)
check("Hopf is supercritical (1 = l1 < 0)", float(l1 < 0), 1, 0)
# simulation and collocation are independent: their periods must agree
k = np.argmin(abs(T - 6))
check("period at taui = 6: simulation vs AUTO", per[k], np.interp(6, px, [p.period for p in po]), 0.05)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
