"""
Morris-Lecar: one- and two-parameter bifurcation diagrams
=========================================================

The XPP Studio "figures" workflow for the Morris-Lecar neuron (Hopf regime:
gca = 4.4, phi = 0.04, v3 = 2, v4 = 30), done in code: the phase plane at
i = 60, the equilibria in i with the periodic orbits born at the first Hopf
point, and that Hopf point followed in two parameters (i, phi).

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. draws the phase plane at i = 60: direction field, nullclines, the
     stable equilibrium and trajectories (figure 1)
  3. follows the equilibria in i with AUTO, computes the first Lyapunov
     coefficient l1 at HB1 and the periodic orbits from HB1 (figure 2)
  4. follows HB1 in (i, phi) and marks where l1 changes sign (generalised
     Hopf points, GH) (figure 3)
  5. checks the equilibrium and the whole Hopf curve against exact formulas

Run:   python 16_2D-FLOWS_morris-lecar-bifurcation.py
Figures go to figures/16_2D-FLOWS_morris-lecar-bifurcation/.

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
    NAME = "16_2D-FLOWS_morris-lecar-bifurcation"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model (t in ms, v in mV)
P = dict(i=60, c=20, vk=-84, vl=-60, vca=120, gk=8, gl=2, gca=4.4, v1=-1.2, v2=18, v3=2, v4=30, phi=0.04)
mb = (ModelBuilder("morris_lecar_figures")
      .comment("Morris-Lecar neuron, Hopf regime (Rinzel & Ermentrout)")
      .function("minf(v)", ".5*(1+tanh((v-v1)/v2))")
      .function("winf(v)", ".5*(1+tanh((v-v3)/v4))")
      .function("tauw(v)", "1/cosh((v-v3)/(2*v4))")
      .ode("v", "(i+gl*(vl-v)+gk*w*(vk-v)+gca*minf(v)*(vca-v))/c")
      .ode("w", "phi*(winf(v)-w)/tauw(v)")
      .aux("ica", "gca*minf(v)*(v-vca)")
      .parameter(**P)
      .initial(v=-60, w=0.015)
      .option(total=400, dt=0.25, meth="rk4", bound=10000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
m.auto.parameters("i", "phi", "gca", "c", "gk", "gl", "v3", "v4")   # phi must be there for step 4
m.buttons.numerics.sing_pt_ctrl(500, 1e-10, 1e-8)   # accurate Newton and eigenvalues

# %% 2. figure 1: phase plane at i = 60
win = (-75, 45, -0.1, 0.55)
nc = m.nullclines(x="v", y="w", xlo=win[0], xhi=win[1], ylo=win[2], yhi=win[3], mesh=200)
X, Y, U, V = m.vector_field("v", "w", *win, grid=21)
eq = m.equilibrium({"v": -37, "w": 0.13})
print("equilibrium at i = 60:", eq, " eigenvalues", np.round(eq.eigenvalues, 5))

fig, ax = plt.subplots(figsize=(6.5, 5))
sx, sy = (win[1] - win[0]), (win[3] - win[2])      # scale arrows in window units
S = np.hypot(U / sx, V / sy)
ax.quiver(X, Y, U / sx / S, V / sy / S, S, cmap="Greys", scale=30, width=0.003, angles="xy")
ax.plot(nc["V"][:, 0], nc["V"][:, 1], color="tab:orange", lw=2, label="v' = 0")
ax.plot(nc["W"][:, 0], nc["W"][:, 1], color="tab:green", lw=2, label="w' = 0")
for (v0, w0), col in zip([(-60, 0.015), (-20, 0.0), (0, 0.45), (30, 0.2)],
                         ["tab:blue", "tab:purple", "tab:red", "tab:brown"]):
    tr = m.run(keep=False, v=v0, w=w0, total=400)
    ax.plot(tr["v"], tr["w"], lw=1.3, color=col)
    ax.plot(v0, w0, "k.", ms=6)
ax.plot(eq.x["V"], eq.x["W"], "ko", ms=8, label="stable equilibrium")
ax.set(xlim=win[:2], ylim=win[2:], xlabel="v (mV)", ylabel="w", title="Morris-Lecar, i = 60")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ml-phaseplane.png", dpi=130)

# %% 3. figure 2: equilibria in i and the orbits from HB1
m.set(i=0)
m.run(total=1000)                # rest at i = 0
m.run_last(total=10)
eqs = m.auto.run("i", start="steady", ds=0.5, dsmax=2, parmin=0, parmax=300, nmax=2000,
                 epsl=1e-6, epsu=1e-6, epss=1e-6)
hb1, hb2 = sorted(eqs.special("HB"), key=lambda p: p.x)
l1, om = m.auto.lyapunov(hb1)
print(f"\nHB1 at i = {hb1.x:.5f}: l1 = {l1:+.5f} ({'sub' if l1 > 0 else 'super'}critical); "
      f"HB2 at i = {hb2.x:.5f}")
po = list(m.auto.run(grab=hb1.label, choice="periodic", ntst=100, ds=0.2, dsmax=1, nmax=300))
po = po[:next(k for k, p in enumerate(po) if p.type == "BP") + 1]   # one pass, HB1 to HB2
print("folds of periodic orbits at i =", sorted({round(p.x, 4) for p in po if p.type == "LP"}))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
xs = np.array([p.x for p in eqs])
vs = np.array([p.u0["v"] for p in eqs])
st = np.array([bool(p.stable) for p in eqs])
ax.plot(xs, np.where(st, vs, np.nan), "r-", lw=2.5, label="stable equilibria")
ax.plot(xs, np.where(~st, vs, np.nan), "k-", lw=1, label="unstable equilibria")
px = np.array([p.x for p in po])
pst = np.array([bool(p.stable) for p in po])
for f in ("umax", "umin"):
    ys = np.array([getattr(p, f)["v"] for p in po])
    ax.plot(px[pst], ys[pst], "g.", ms=3)
    ax.plot(px[~pst], ys[~pst], ".", color="tab:blue", ms=3)
ax.plot([], [], "g.", label="stable orbits (max/min v)")
ax.plot([], [], ".", color="tab:blue", label="unstable orbits")
ax.plot([hb1.x, hb2.x], [hb1.u0["v"], hb2.u0["v"]], "o", color="purple", ms=8, label="Hopf")
ax.set(xlim=(0, 300), xlabel="i", ylabel="v (mV)", title=f"Equilibria and orbits in i (l1 at HB1 = {l1:.4f})")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "ml-bifurcation.png", dpi=130)

# %% 4. figure 3: HB1 in (i, phi)
c1 = m.auto.run(grab=hb1.label, choice="two param", param2="phi", ds=0.5, dsmax=1, parmin=0, parmax=300,
                nmax=1000)
c2 = m.auto.run(grab=hb1.label, choice="two param", ds=-0.5)
gh = m.auto.gh_points(c1) + m.auto.gh_points(c2)
hopf = np.array([[q.x, q.par["phi"], q.u0["v"]] for c in (c1, c2) for q in c])
print(f"Hopf curve: {len(hopf)} points, phi up to {hopf[:, 1].max():.4f}; GH points:",
      [(round(g["par"], 2), round(g["par2"], 4)) for g in gh])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for c in (c1, c2):
    ax.plot([q.x for q in c], [q.par["phi"] for q in c], color="tab:red", lw=2)
ax.plot([g["par"] for g in gh], [g["par2"] for g in gh], "D", color="purple", ms=7, label="GH (l1 = 0)")
ax.plot(hb1.x, P["phi"], "ko", ms=6, label="HB1 at phi = 0.04")
ax.text(150, 0.12, "rest unstable\n(inside the Hopf curve)", ha="center", fontsize=9)
ax.set(xlim=(60, 250), ylim=(0, 0.42), xlabel="i", ylabel="phi", title="Hopf curve in (i, phi)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "ml-twopar.png", dpi=130)

# %% 5. checks
# exact: on an equilibrium w = winf(v), so i(v) is explicit; the trace of the Jacobian is
# a11(v) - phi/tauw(v), so the Hopf curve is (i(v), phi(v) = a11(v) tauw(v)), with det > 0
th = lambda v, a, b: np.tanh((v - a) / b)
minf = lambda v: 0.5 * (1 + th(v, P["v1"], P["v2"]))
winf = lambda v: 0.5 * (1 + th(v, P["v3"], P["v4"]))
tauw = lambda v: 1 / np.cosh((v - P["v3"]) / (2 * P["v4"]))
i_of = lambda v: -(P["gl"] * (P["vl"] - v) + P["gk"] * winf(v) * (P["vk"] - v)
                   + P["gca"] * minf(v) * (P["vca"] - v))
a11 = lambda v: (-P["gl"] - P["gk"] * winf(v) - P["gca"] * minf(v)
                 + P["gca"] * (1 - th(v, P["v1"], P["v2"])**2) / (2 * P["v2"]) * (P["vca"] - v)) / P["c"]
phi_of = lambda v: a11(v) * tauw(v)
vg = np.linspace(-60, 40, 200001)
print("\nchecks:")
v60 = vg[np.argmin(abs(i_of(vg) - 60))]
check("equilibrium v at i = 60 (exact, grid 5e-4 mV)", eq.x["V"], v60, 1e-3)
check("equilibrium w = winf(v)", eq.x["W"], winf(eq.x["V"]), 1e-8)
check("Hopf curve: largest |i_AUTO - i(v)|", np.abs(hopf[:, 0] - i_of(hopf[:, 2])).max(), 0, 1e-3)
check("Hopf curve: largest |phi_AUTO - phi(v)|", np.abs(hopf[:, 1] - phi_of(hopf[:, 2])).max(), 0, 1e-6)
check("top of the Hopf curve: largest phi (exact)", hopf[:, 1].max(), phi_of(vg).max(), 1e-4)
# exact: as phi -> 0 the Hopf points tend to the knees of the v-nullcline, where a11 = 0
knees = sorted(i_of(vg[1:][np.sign(a11(vg[:-1])) != np.sign(a11(vg[1:]))]))
ends = sorted(q[-1].x for q in (c1, c2))
check("Hopf curve end as phi -> 0, left knee (exact)", ends[0], knees[0], 0.01)
check("Hopf curve end as phi -> 0, right knee (exact)", ends[1], knees[1], 0.01)
check("HB1 is subcritical at phi = 0.04 (1 = l1 > 0)", float(l1 > 0), 1, 0)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
