"""
Morris-Lecar: a neuron with a Hopf onset (class II)
===================================================

The Morris-Lecar model of a barnacle muscle fibre in Rinzel and Ermentrout's
parameters: a calcium current with instant activation minf(v) and a potassium
current with recovery variable w.  With phi = 0.04 and gca = 4.4 repetitive
firing starts at a Hopf bifurcation as the applied current i grows, and it
starts at a nonzero frequency: a class II neuron in Hodgkin's sense.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. shows voltage traces below, inside and above the firing range
  3. measures the firing rate against i (an f-I curve) from simulations, once
     sweeping i up from rest and once sweeping down from firing
  4. follows the equilibria in i with AUTO (two Hopf points, both
     subcritical) and the periodic orbits between them; their rate 1000/T
     lies on the simulated f-I curve, and the fold of orbits explains the
     hysteresis
  5. checks the Hopf points and onset frequencies against exact values

Run:   python 2D-FLOWS_morris-lecar.py
Figures go to figures/2D-FLOWS_morris-lecar/.

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
    NAME = "2D-FLOWS_morris-lecar"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model (t in ms, v in mV)
P = dict(i=60, c=20, vk=-84, vl=-60, vca=120, gk=8, gl=2, gca=4.4, v1=-1.2, v2=18, v3=2, v4=30, phi=0.04)
mb = (ModelBuilder("morris_lecar")
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


def rate(tr):
    """spikes per second from upward crossings of v = 0 (t in ms)"""
    v = tr["v"]
    up = tr.t[1:][(v[:-1] < 0) & (v[1:] >= 0)]
    return 1000 / np.diff(up).mean() if len(up) > 2 else 0.0


# %% 2. voltage traces
fig, ax = plt.subplots(figsize=(7, 3.5))
for i_, col in ((60, "tab:blue"), (100, "tab:red"), (230, "tab:purple")):
    tr = m.run(keep=False, i=i_, total=500)
    ax.plot(tr.t, tr["v"], color=col, lw=1.2, label=f"i = {i_}")
ax.set(xlabel="t (ms)", ylabel="v (mV)", ylim=(-65, 80), title="Rest, repetitive firing, depolarised rest")
ax.legend(fontsize=8, loc="upper right", ncol=3)
fig.tight_layout()
fig.savefig(FIG / "traces.png", dpi=130)

# %% 3. f-I curve from simulations: sweep up from rest, then down from firing
I = np.arange(80.0, 240.1, 1.0)
state = {"v": -60, "w": 0.015}
f_up, f_down = [], []
for i_ in I:                                         # each run starts where the last one ended
    tr = m.run(keep=False, i=i_, total=2000, trans=1000, **state)
    f_up.append(rate(tr))
    state = {"v": tr["v"][-1], "w": tr["w"][-1]}
for i_ in I[::-1]:
    tr = m.run(keep=False, i=i_, total=2000, trans=1000, **state)
    f_down.append(rate(tr))
    state = {"v": tr["v"][-1], "w": tr["w"][-1]}
f_up, f_down = np.array(f_up), np.array(f_down[::-1])
on_up = I[np.argmax(f_up > 0)]                       # first current that fires on the way up
on_down = I[np.argmax(f_down > 0)]                   # lowest current still firing on the way down
print(f"\nsweeping up: firing from i = {on_up:g} at {f_up[f_up > 0][0]:.2f} Hz;"
      f" sweeping down: firing down to i = {on_down:g} at {f_down[f_down > 0][0]:.2f} Hz")

# %% 4. AUTO: equilibria in i, Hopf points, orbits between them
m.set(i=0)
m.run(total=1000)                # rest at i = 0
m.run_last(total=10)
eqs = m.auto.run("i", start="steady", ds=0.5, dsmax=2, parmin=0, parmax=300, nmax=2000,
                 epsl=1e-6, epsu=1e-6, epss=1e-6)        # tighter than AUTO's 1e-4: Hopf points to 1e-5
hbs = eqs.special("HB")
for p in hbs:
    l1, om = m.auto.lyapunov(p)
    print(f"Hopf at i = {p.x:.4f}: l1 = {l1:+.5f} ({'sub' if l1 > 0 else 'super'}critical), "
          f"frequency at onset {1000 * om / (2 * np.pi):.3f} Hz")
po = list(m.auto.run(grab=hbs[0].label, choice="periodic", ntst=100, ds=0.2, dsmax=1, nmax=300))
po = po[:next(k for k, p in enumerate(po) if p.type == "BP") + 1]   # one pass, HB1 to HB2
lps = sorted({round(p.x, 4) for p in po if p.type == "LP"})
print("folds of periodic orbits at i =", lps)
pst = [p for p in po if p.stable]
lp1 = min((p for p in po if p.type == "LP"), key=lambda p: p.x)
print(f"orbit at the lower fold: period {lp1.period:.2f} ms, {1000 / lp1.period:.2f} Hz")

fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(I, f_up, "o", color="tab:blue", ms=3.5, label="simulation, i increasing")
ax.plot(I, f_down, "x", color="tab:red", ms=4, label="simulation, i decreasing")
pun = [p for p in po if not p.stable]
ax.plot([p.x for p in pun], [1000 / p.period for p in pun], ".", color="0.6", ms=3, label="AUTO, unstable orbits")
ax.plot([p.x for p in pst], [1000 / p.period for p in pst], ".", color="tab:green", ms=3, label="AUTO, stable orbits")
for p in hbs:
    ax.axvline(p.x, color="purple", ls=":", lw=1)
ax.text(hbs[0].x + 1, 1, "HB1", color="purple")
ax.text(hbs[1].x + 1, 1, "HB2", color="purple")
ax.set(xlabel="i", ylabel="firing rate (spikes / s)", xlim=(80, 240), ylim=(-1, 26),
       title="f-I curve: firing starts at a nonzero rate (class II)")
ax.legend(fontsize=8, loc="lower center")
fig.tight_layout()
fig.savefig(FIG / "f_I_curve.png", dpi=130)

# %% 5. checks
# exact: on an equilibrium w = winf(v), so i is explicit in v; the Hopf points are where the
# trace of the Jacobian vanishes, and the frequency there is sqrt(det)/(2 pi)
th = lambda v, a, b: np.tanh((v - a) / b)
minf = lambda v: 0.5 * (1 + th(v, P["v1"], P["v2"]))
winf = lambda v: 0.5 * (1 + th(v, P["v3"], P["v4"]))
tauw = lambda v: 1 / np.cosh((v - P["v3"]) / (2 * P["v4"]))
i_of = lambda v: -(P["gl"] * (P["vl"] - v) + P["gk"] * winf(v) * (P["vk"] - v)
                   + P["gca"] * minf(v) * (P["vca"] - v))
a11 = lambda v: (-P["gl"] - P["gk"] * winf(v) - P["gca"] * minf(v)
                 + P["gca"] * (1 - th(v, P["v1"], P["v2"])**2) / (2 * P["v2"]) * (P["vca"] - v)) / P["c"]
a22 = lambda v: -P["phi"] / tauw(v)
a12 = lambda v: P["gk"] * (P["vk"] - v) / P["c"]
a21 = lambda v: P["phi"] * (1 - th(v, P["v3"], P["v4"])**2) / (2 * P["v4"]) / tauw(v)
tr_ = lambda v: a11(v) + a22(v)
vg = np.linspace(-60, 40, 100001)
vh = []
for k in np.where(np.sign(tr_(vg[:-1])) != np.sign(tr_(vg[1:])))[0]:
    a, b = vg[k], vg[k + 1]
    for _ in range(60):                               # bisection
        c = 0.5 * (a + b)
        a, b = (c, b) if np.sign(tr_(c)) == np.sign(tr_(a)) else (a, c)
    vh.append(0.5 * (a + b))
print("\nchecks:")
for p, v in zip(hbs, vh):
    k = 1 if p is hbs[0] else 2
    check(f"HB{k} at i (exact, trace = 0)", p.x, i_of(v), 1e-3)
    om_exact = np.sqrt(a11(v) * a22(v) - a12(v) * a21(v))
    check(f"HB{k} frequency in Hz (exact)", 1000 * m.auto.lyapunov(p)[1] / (2 * np.pi),
          1000 * om_exact / (2 * np.pi), 1e-3)
check("HB1 is subcritical (1 = l1 > 0)", float(m.auto.lyapunov(hbs[0])[0] > 0), 1, 0)
# the simulations jump to firing at HB1 going up, and keep firing down to the fold of orbits
# (just above HB1 the focus is only weakly unstable: at i = 94 the eigenvalues have real part
# 5e-4 per ms, so leaving rest takes many seconds; the sweep fires from the next grid points)
check("no firing below HB1 sweeping up (count)", np.sum((f_up > 0) & (I < hbs[0].x)), 0, 0)
check("last firing sweeping down = fold of orbits (grid)", on_down, np.ceil(lp1.x), 0)
# class II: the lowest rate is that of the stable orbit at the fold, far from zero
sx, sf = np.array([[p.x, 1000 / p.period] for p in sorted(pst, key=lambda p: p.x) if p.x < 200]).T
check("lowest simulated rate vs AUTO stable orbit there (Hz)", f_down[f_down > 0][0],
      np.interp(on_down, sx, sf), 0.02)
check("rate at i = 150: simulation vs AUTO (Hz)", f_up[I == 150][0], np.interp(150, sx, sf), 0.02)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
