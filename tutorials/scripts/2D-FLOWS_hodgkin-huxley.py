"""
Hodgkin-Huxley: the squid giant axon
====================================

The four-variable Hodgkin-Huxley model (v in mV, t in ms, i in uA/cm^2):
sodium activation m and inactivation h, potassium activation n.  At i = 10
it fires repeatedly.  Repetitive firing begins at a fold of periodic orbits
near i = 6.26, below the subcritical Hopf point at i = 9.78, so between the
two the cell can rest or fire depending on how it starts.  The (v, n) plane
here is only a projection of the 4-D phase space: there are no nullclines.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. finds the period at i = 10 with XPP's Initialconds > Periodic
  3. follows the rest state in i with AUTO (Hopf points near 9.78 and 154.5)
     and the periodic orbits from the first Hopf point, with their folds
  4. shows bistability at i = 8: a small kick dies out, a larger one starts
     repetitive firing
  5. checks the Hopf points against an exact linearisation and the rest
     against values from the literature

Run:   python 2D-FLOWS_hodgkin-huxley.py
Figures go to figures/2D-FLOWS_hodgkin-huxley/.

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
    NAME = "2D-FLOWS_hodgkin-huxley"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
P = dict(i=10, gna=120, gk=36, gl=0.3, ena=50, ek=-77, el=-54.4, cm=1)
mb = (ModelBuilder("hodgkin_huxley")
      .comment("Hodgkin-Huxley model of the squid giant axon: v in mV, t in ms, i in uA/cm^2")
      .function("am(v)", "if(abs(v+40)<1e-7)then(1)else(0.1*(v+40)/(1-exp(-(v+40)/10)))")
      .function("bm(v)", "4*exp(-(v+65)/18)")
      .function("ah(v)", "0.07*exp(-(v+65)/20)")
      .function("bh(v)", "1/(1+exp(-(v+35)/10))")
      .function("an(v)", "if(abs(v+55)<1e-7)then(0.1)else(0.01*(v+55)/(1-exp(-(v+55)/10)))")
      .function("bn(v)", "0.125*exp(-(v+65)/80)")
      .ode("v", "(i-gna*m^3*h*(v-ena)-gk*n^4*(v-ek)-gl*(v-el))/cm")
      .ode("m", "am(v)*(1-m)-bm(v)*m")
      .ode("h", "ah(v)*(1-h)-bh(v)*h")
      .ode("n", "an(v)*(1-n)-bn(v)*n")
      .parameter(**P)
      .initial(v=-65, m=0.05, h=0.6, n=0.32)
      .option(total=300, dt=0.02, meth="rk4"))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. the period at i = 10 (Initialconds > Periodic: one period of the stable orbit)
m.run(total=300, trans=0)
m.buttons.initialconds.periodic()
T10 = m.numerics["total"]
print(m.transcript.strip().splitlines()[-1])
m.set(total=300)

# %% 3. AUTO: rest state in i, Hopf points, periodic orbits
m.set(i=0)
m.run(total=300)                 # the rest state at i = 0, below both Hopf points
m.run_last(total=10)
eqs = m.auto.run("i", start="steady", ds=0.2, dsmax=0.8, parmin=0, parmax=200, nmax=2000)
hb1, hb2 = sorted(eqs.special("HB"), key=lambda p: p.x)
l1, om = m.auto.lyapunov(hb1)
print(f"\nHopf points at i = {hb1.x:.4f} (l1 = {l1:+.4f}) and {hb2.x:.4f}")
po = list(m.auto.run(grab=hb1.label, choice="periodic", ntst=60, ds=0.05, dsmax=0.5, nmax=400))
po = po[:next(k for k, p in enumerate(po) if p.type == "BP") + 1]   # up to the second Hopf point
lps = sorted(p.x for p in po if p.type == "LP")
print("folds of periodic orbits at i =", np.round(lps, 4))
stab = sorted([(p.x, p.period) for p in po if p.stable and p.x < 140])
sx, sT = np.array(stab).T                            # the stable branch (period against i)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
xs = np.array([p.x for p in eqs])
vs = np.array([p.u0["v"] for p in eqs])
st = np.array([bool(p.stable) for p in eqs])
px = np.array([p.x for p in po])
pst = np.array([bool(p.stable) for p in po])
for ax in axs:
    ax.plot(xs, np.where(st, vs, np.nan), "r-", lw=2.5, label="stable rest")
    ax.plot(xs, np.where(~st, vs, np.nan), "k-", lw=1, label="unstable rest")
    for f in ("umax", "umin"):
        ys = np.array([getattr(p, f)["v"] for p in po])
        ax.plot(px[pst], ys[pst], "g.", ms=3)
        ax.plot(px[~pst], ys[~pst], ".", color="tab:blue", ms=3)
    ax.plot([hb1.x, hb2.x], [hb1.u0["v"], hb2.u0["v"]], "o", color="purple", ms=7, label="Hopf")
    ax.set(xlabel="i (uA/cm^2)", ylabel="v (mV)")
axs[0].plot([], [], "g.", label="stable orbits (max/min v)")
axs[0].plot([], [], ".", color="tab:blue", label="unstable orbits")
axs[0].set(xlim=(0, 200), title="Rest and repetitive firing")
axs[0].legend(fontsize=8, loc="center right")
axs[1].axvspan(lps[0], hb1.x, color="0.92", zorder=0)
axs[1].set(xlim=(5, 11), title="Zoom: bistable between the fold and HB1")
fig.tight_layout()
fig.savefig(FIG / "bifurcation.png", dpi=130)

# %% 4. bistability at i = 8
m.set(i=8)
rest = m.equilibrium({"v": -62, "m": 0.07, "h": 0.5, "n": 0.36})
r0 = {k.lower(): v for k, v in rest.x.items()}
quiet = m.run(keep=False, total=300, **{**r0, "v": r0["v"] + 2})   # a 2 mV kick from rest
kick = m.run(keep=False, total=300, **{**r0, "v": r0["v"] + 10})   # a 10 mV kick from rest
late = kick.t > 150
up = kick.t[1:][(kick["v"][:-1] < 0) & (kick["v"][1:] >= 0)]
T8 = np.diff(up[up > 150]).mean()
print(f"\ni = 8: rest at v = {rest.x['V']:.3f} mV (stable: {rest.stable}); kicked cell fires, period {T8:.3f} ms")

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
axs[0].plot(quiet.t, quiet["v"], color="tab:blue", lw=1.5, label="start 2 mV above rest")
axs[0].plot(kick.t, kick["v"], color="tab:red", lw=1, label="start 10 mV above rest")
axs[0].set(xlabel="t (ms)", ylabel="v (mV)", ylim=(-85, 65), title="i = 8: rest and firing coexist")
axs[0].legend(fontsize=8, loc="upper right", ncol=2)
axs[1].plot(kick["v"][late], kick["n"][late], color="tab:red", lw=1.5, label="firing (stable orbit)")
axs[1].plot(rest.x["V"], rest.x["N"], "o", color="tab:blue", ms=8, label="rest (stable)")
axs[1].set(xlabel="v (mV)", ylabel="n", xlim=(-85, 55), ylim=(0.2, 0.85), title="projection on (v, n)")
axs[1].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "bistability.png", dpi=130)

# %% 5. checks
# exact: at rest each gate is at its steady state x = a/(a+b), so i is explicit in v; the
# Jacobian is written out by hand (rate derivatives by central differences, error ~1e-10),
# and a Hopf point is where its leading complex pair crosses the imaginary axis
am = lambda v: 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10))
bm = lambda v: 4 * np.exp(-(v + 65) / 18)
ah = lambda v: 0.07 * np.exp(-(v + 65) / 20)
bh = lambda v: 1 / (1 + np.exp(-(v + 35) / 10))
an = lambda v: 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10))
bn = lambda v: 0.125 * np.exp(-(v + 65) / 80)
gates = [(am, bm), (ah, bh), (an, bn)]
inf = lambda v: [a(v) / (a(v) + b(v)) for a, b in gates]
d = lambda f, v: (f(v + 1e-5) - f(v - 1e-5)) / 2e-5


def i_of(v):
    mm, hh, nn = inf(v)
    return P["gna"] * mm**3 * hh * (v - P["ena"]) + P["gk"] * nn**4 * (v - P["ek"]) + P["gl"] * (v - P["el"])


def growth(v):                    # largest real part of the eigenvalues at rest
    mm, hh, nn = inf(v)
    J = np.zeros((4, 4))
    J[0] = [-(P["gna"] * mm**3 * hh + P["gk"] * nn**4 + P["gl"]), -3 * P["gna"] * mm**2 * hh * (v - P["ena"]),
            -P["gna"] * mm**3 * (v - P["ena"]), -4 * P["gk"] * nn**3 * (v - P["ek"])]
    for k, ((a, b), x) in enumerate(zip(gates, (mm, hh, nn))):
        J[k + 1, 0] = d(a, v) * (1 - x) - d(b, v) * x
        J[k + 1, k + 1] = -(a(v) + b(v))
    return np.linalg.eigvals(J / P["cm"]).real.max()


def hopf_i(lo, hi):               # bisection on the sign of growth(v)
    for _ in range(60):
        c = 0.5 * (lo + hi)
        lo, hi = (c, hi) if np.sign(growth(c)) == np.sign(growth(lo)) else (lo, c)
    return i_of(0.5 * (lo + hi))


print("\nchecks:")
check("HB1 (exact linearisation)", hb1.x, hopf_i(hb1.u0["v"] - 1, hb1.u0["v"] + 1), 1e-3)
check("HB2 (exact linearisation)", hb2.x, hopf_i(hb2.u0["v"] - 1, hb2.u0["v"] + 1), 1e-3)
check("rest at i = 8 satisfies i(v) = 8 (exact)", i_of(rest.x["V"]), 8, 1e-6)
check("HB1 is subcritical (1 = l1 > 0; Hassard 1978)", float(l1 > 0), 1, 0)
check("period at i = 10, Periodic (XPP Studio: 14.64 ms)", T10, 14.64, 0.005)
check("period at i = 10: Periodic vs AUTO orbit", T10, np.interp(10, sx, sT), 1e-3)
check("lowest fold of orbits (Rinzel & Miller 1980: 6.26)", lps[0], 6.26, 0.01)
check("i = 8, 2 mV kick: no spike (largest v below 0, 1 = yes)", float(quiet["v"].max() < 0), 1, 0)
check("i = 8, 2 mV kick: back at rest at t = 300 (mV)", quiet["v"][-1], rest.x["V"], 1e-3)
check("i = 8, kicked start: period vs AUTO stable orbit", T8, np.interp(8, sx, sT), 0.02)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
