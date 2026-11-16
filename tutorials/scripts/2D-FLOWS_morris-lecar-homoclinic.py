"""
Morris-Lecar with a saddle: a homoclinic orbit
==============================================

With gca = 4, phi = 0.23, v3 = 12 and v4 = 17.4 the Morris-Lecar model has
three equilibria over a range of i: a stable node (rest), a saddle and a
focus.  The periodic orbit born at the Hopf point of the focus grows until it
touches the saddle at i = 35.0067: a homoclinic bifurcation.  Close to it the
orbit spends longer and longer near the saddle, and its period grows like
    T ~ -(1/lambda_u) ln(i - i_hom),
where lambda_u is the saddle's unstable eigenvalue.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. follows the equilibria in i with AUTO: a fold and a Hopf point
  3. follows the periodic orbits from the Hopf point down to the homoclinic
     orbit and plots their period (log scale, and against ln(i - i_hom))
  4. fits the logarithmic law and compares the slope with 1/lambda_u
  5. draws the phase plane at i = 35.01 with the long-period orbit passing
     close to the saddle, from a plain simulation
  6. checks the fold, the Hopf point and the period law

Run:   python 2D-FLOWS_morris-lecar-homoclinic.py
Figures go to figures/2D-FLOWS_morris-lecar-homoclinic/.

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

HERE = Path(__file__).resolve().parent
NAME = Path(__file__).stem
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model (t in ms, v in mV)
P = dict(i=37, c=20, vk=-84, vl=-60, vca=120, gk=8, gl=2, gca=4, v1=-1.2, v2=18, v3=12, v4=17.4, phi=0.23)
mb = (ModelBuilder("morris_lecar_homoclinic")
      .comment("Morris-Lecar with a saddle: a homoclinic orbit of the saddle")
      .function("minf(v)", ".5*(1+tanh((v-v1)/v2))")
      .function("winf(v)", ".5*(1+tanh((v-v3)/v4))")
      .function("tauw(v)", "1/cosh((v-v3)/(2*v4))")
      .ode("v", "(i+gl*(vl-v)+gk*w*(vk-v)+gca*minf(v)*(vca-v))/c")
      .ode("w", "phi*(winf(v)-w)/tauw(v)")
      .parameter(**P)
      .initial(v=-60, w=0.015)
      .option(total=400, dt=0.25, bound=10000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
m.buttons.numerics.sing_pt_ctrl(500, 1e-10, 1e-8)   # accurate Newton and eigenvalues

# %% 2. AUTO: equilibria in i
m.run(total=2000)                # rest on the stable node at i = 37
m.run_last(total=10)
eqs = m.auto.run("i", start="steady", ds=0.1, dsmax=0.5, parmin=-30, parmax=60, nmax=3000,
                 epsl=1e-6, epsu=1e-6, epss=1e-6)
lp = [p for p in eqs.special("LP") if p.x > 0][0]
hb = eqs.special("HB")[0]
l1, om = m.auto.lyapunov(hb)
print(f"fold (saddle-node) at i = {lp.x:.5f}; Hopf at i = {hb.x:.5f}, l1 = {l1:+.5f}")

# %% 3. AUTO: periodic orbits from the Hopf point to the homoclinic orbit
po = m.auto.run(grab=hb.label, choice="periodic", ntst=100, ds=-0.02, dsmax=0.2, nmax=600)
x = np.array([p.x for p in po])
T = np.array([p.period for p in po])
k_lp = next(k for k, p in enumerate(po) if p.type == "LP")    # the branch turns at a fold of orbits
good = (T < 1e4) & (np.arange(len(po)) > k_lp)       # keep the way down; drop AUTO's last points
i_hom = x[good].min()                                # the orbit with the longest period is there
print(f"{len(po)} orbits; longest period {T[good].max():.0f} ms; homoclinic at i = {i_hom:.7f}")

# %% 4. the logarithmic law
sad = m.equilibrium({"v": -22, "w": 0.02})          # the saddle at i = 37 ...
m.set(i=i_hom)
sad = m.equilibrium(sad.x)                           # ... followed to i_hom
lam_u = sad.eigenvalues.real.max()
lam_s = sad.eigenvalues.real.min()
print(f"saddle at i_hom: v = {sad.x['V']:.4f}, eigenvalues {lam_u:.5f}, {lam_s:.5f}")
# fit T = C - b ln(i - i_hom) where the law holds and i - i_hom is well resolved (80 < T < 250)
fit = good & (T > 80) & (T < 250) & (x > i_hom)
b, C = np.polyfit(-np.log(x[fit] - i_hom), T[fit], 1)
print(f"fit: T = {C:.3f} - {b:.4f} ln(i - i_hom)   (1/lambda_u = {1 / lam_u:.4f})")

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
axs[0].semilogy(x[:k_lp + 1], T[:k_lp + 1], ".", color="0.6", ms=3, label="up from the Hopf point")
axs[0].semilogy(x[good], T[good], ".", color="tab:blue", ms=3, label="down to the homoclinic")
axs[0].axvline(i_hom, color="purple", ls=":", label=f"homoclinic, i = {i_hom:.4f}")
axs[0].axvline(hb.x, color="tab:red", ls=":", label=f"Hopf, i = {hb.x:.3f}")
axs[0].set(xlabel="i", ylabel="period (ms)", title="Orbits from the Hopf point")
axs[0].legend(fontsize=8)
d = np.logspace(-11, 0, 50)
s = good & (x > i_hom)
axs[1].semilogx(x[s] - i_hom, T[s], ".", color="tab:blue", ms=3, label="AUTO")
axs[1].semilogx(d, C - np.log(d) / lam_u, "-", color="tab:orange", lw=1.5,
                label="slope 1/lambda_u of the saddle")
axs[1].axvspan((x[fit] - i_hom).min(), (x[fit] - i_hom).max(), color="0.9", zorder=0, label="fitted range")
axs[1].set(xlabel="i - i_hom", ylabel="period (ms)", ylim=(0, 400), title="T against ln(i - i_hom)")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "period.png", dpi=130)

# %% 5. phase plane at i = 35.01: the orbit passes close to the saddle
m.set(i=35.01)
win = (-60, 40, -0.05, 0.5)
nc = m.nullclines(x="v", y="w", xlo=win[0], xhi=win[1], ylo=win[2], yhi=win[3], mesh=300)
cyc = m.run(keep=False, v=0, w=0.3, total=3000, trans=2000)
vv = cyc["v"]
ups = cyc.t[1:][(vv[:-1] < 0) & (vv[1:] >= 0)]
T_sim = np.diff(ups).mean()
eq3 = [m.equilibrium(g) for g in ({"v": -36, "w": 0.004}, {"v": -22, "w": 0.02}, {"v": 4, "w": 0.3})]
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(nc["V"][:, 0], nc["V"][:, 1], color="tab:orange", lw=2, label="v' = 0")
ax.plot(nc["W"][:, 0], nc["W"][:, 1], color="tab:green", lw=2, label="w' = 0")
ax.plot(cyc["v"], cyc["w"], color="tab:blue", lw=1.5, label=f"stable orbit, period {T_sim:.1f} ms")
for e, name in zip(eq3, ("stable node", "saddle", "unstable focus")):
    ax.plot(e.x["V"], e.x["W"], "o", mfc="k" if e.stable else "white", mec="k", ms=8)
    ax.annotate(name, (e.x["V"], e.x["W"]), xytext=(6, -12), textcoords="offset points", fontsize=8)
ax.set(xlim=win[:2], ylim=win[2:], xlabel="v (mV)", ylabel="w", title="i = 35.01, just above the homoclinic")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "phase_plane.png", dpi=130)

# %% 6. checks
# exact: on an equilibrium w = winf(v), so i(v) is explicit; folds where di/dv = 0, Hopf
# where the trace a11(v) - phi/tauw(v) of the Jacobian vanishes
th = lambda v, a, b: np.tanh((v - a) / b)
minf = lambda v: 0.5 * (1 + th(v, P["v1"], P["v2"]))
winf = lambda v: 0.5 * (1 + th(v, P["v3"], P["v4"]))
tauw = lambda v: 1 / np.cosh((v - P["v3"]) / (2 * P["v4"]))
i_of = lambda v: -(P["gl"] * (P["vl"] - v) + P["gk"] * winf(v) * (P["vk"] - v)
                   + P["gca"] * minf(v) * (P["vca"] - v))
a11 = lambda v: (-P["gl"] - P["gk"] * winf(v) - P["gca"] * minf(v)
                 + P["gca"] * (1 - th(v, P["v1"], P["v2"])**2) / (2 * P["v2"]) * (P["vca"] - v)) / P["c"]
vg = np.linspace(-60, 40, 400001)
ig = i_of(vg)
print("\nchecks:")
check("fold of equilibria (exact, largest i on the S below)", lp.x, ig[(vg > -40) & (vg < -10)].max(), 1e-4)
trc = a11(vg) - P["phi"] / tauw(vg)
k = np.where((np.sign(trc[:-1]) != np.sign(trc[1:])) & (vg[:-1] > 0))[0][0]
check("Hopf point (exact, trace = 0)", hb.x, ig[k], 1e-3)
check("homoclinic i (XPP Studio: 35.0067)", i_hom, 35.0067, 1e-4)
check("slope of T against -ln(i - i_hom) vs 1/lambda_u", b, 1 / lam_u, 0.01 / lam_u)
# Andronov-Leontovich: lambda_u + lambda_s < 0, so the orbits next to the homoclinic orbit
# are stable (AUTO's Floquet multipliers are unreliable there); the simulation shows it
check("saddle quantity lambda_u + lambda_s < 0 (1 = yes)", float(lam_u + lam_s < 0), 1, 0)
check("period at i = 35.01: simulation vs AUTO (ms)", T_sim, np.interp(35.01, x[s][::-1], T[s][::-1]), 0.25)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
