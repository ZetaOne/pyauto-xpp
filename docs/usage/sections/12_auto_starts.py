# %% [markdown]
# ## AUTO IV: many initial conditions, files and buttons
#
# **From many initial conditions.** Integrate from each start, see where it
# ends (rest or an orbit), and start AUTO there. Each start gives a branch;
# they should all lie on one diagram. Hodgkin-Huxley at I = 8, where rest
# and repetitive firing coexist. (The full study, 49 starts, is
# `rungs/AUTO/hh/hh_auto.py`.)

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)
from pyxpp.builder import ModelBuilder

hh = (ModelBuilder("hh")
      .parameter(I=8.0, vna=50, vk=-77, vl=-54.4, gna=120, gk=36, gl=0.3, c=1)
      .function("am(v)", ".1*(v+40)/(1-exp(-(v+40)/10))")
      .function("bm(v)", "4*exp(-(v+65)/18)")
      .function("ah(v)", ".07*exp(-(v+65)/20)")
      .function("bh(v)", "1/(1+exp(-(v+35)/10))")
      .function("an(v)", ".01*(v+55)/(1-exp(-(v+55)/10))")
      .function("bn(v)", ".125*exp(-(v+65)/80)")
      .ode("v", "(I - gna*h*(v-vna)*m^3 - gk*(v-vk)*n^4 - gl*(v-vl))/c")
      .ode("m", "am(v)*(1-m) - bm(v)*m")
      .ode("h", "ah(v)*(1-h) - bh(v)*h")
      .ode("n", "an(v)*(1-n) - bn(v)*n")
      .initial(v=-65, m=0.053, h=0.596, n=0.318)
      .option(total=300, dt=0.01, meth="rk4", maxstor=100000))
m = hh.load("hh.ode")
rest = m.equilibrium({"v": -60, "m": 0.08, "h": 0.5, "n": 0.35})
starts = {"near rest": {k.lower(): x + (0.5 if k == "V" else 0) for k, x in rest.x.items()},
          "v = -40": dict(v=-40, m=0.3, h=0.3, n=0.5),
          "v = 20": dict(v=20, m=0.9, h=0.2, n=0.6)}
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, ic in starts.items():
    m.auto.reset()
    tr = m.run(I=8.0, trans=0, total=300, **ic)
    late = tr.t > 150
    v = tr["v"]
    ax[0].plot(tr.t, v, lw=0.8, label=name)
    if np.ptp(v[late]) < 0.5:                                   # at rest: steady state
        m.set(**{k.lower(): float(tr[k][-1]) for k in m.vars})
        d = m.auto.run("I", start="steady", ntst=15, nmax=300, npr=500, ds=0.1, dsmax=1, parmin=0, parmax=20)
        I = [p.x for p in d]; ax[1].plot(I, [p.u0["v"] for p in d], "-", label=f"{name}: equilibria")
        print(f"{name:9s} -> rest;   AUTO special points:", [(p.type, round(p.x, 4)) for p in d.labeled() if p.type == "HB"])
    else:                                                       # spiking: one period, then periodic
        up = np.where((v[:-1] < 0) & (v[1:] >= 0) & late[1:])[0]
        T = float(tr.t[up[-1]] - tr.t[up[-2]]); k = up[-2] + 1
        m.set(trans=0, total=T, **{n.lower(): float(tr[n][k]) for n in m.vars})
        m.run()
        d = m.auto.run("I", start="periodic", ntst=60, nmax=200, npr=500, ds=-0.05, dsmax=0.5, parmin=0, parmax=20)
        ax[1].plot([p.x for p in d], [p.umax["v"] for p in d], ".", ms=2, label=f"{name}: orbits (max v)")
        print(f"{name:9s} -> spiking, period {T:.3f};  folds of orbits (AUTO passes each twice):", sorted({round(p.x, 4) for p in d.labeled() if p.type == "LP"}))
ax[0].set(xlabel="t", ylabel="v", title="runs at I = 8"); ax[0].legend(fontsize=7)
ax[1].set(xlabel="I", ylabel="v", xlim=(5, 12), title="AUTO started from where each run ended"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "hh_starts.png", dpi=100)

# %% [markdown]
# **Files.** `write_pts` writes XPP's text table of the diagram; `save` and
# `load` keep a whole diagram (`.auto` file) to reload later; `reset`
# deletes the diagram and AUTO's files; `clear_grab` drops the grabbed
# point.

# %%
lm = pyxpp.load(PKG / "examples" / "lecar.ode")
lm.run(total=300); lm.run_last(total=10)
br = lm.auto.run("iapp", start="steady", ds=0.01, dsmax=0.02, parmin=-0.2, parmax=0.5, nmax=2000)
lm.auto.write_pts("diagram.dat")
print((lm.workdir / "diagram.dat").read_text().splitlines()[:3])
lm.auto.save("lecar.auto")
n_before = len(lm.auto.diagram)
lm.auto.reset()
print("after reset:", len(lm.auto.diagram), "points")
lm.auto.load("lecar.auto")
print("after load:", len(lm.auto.diagram), "points (was", n_before, ")")
lm.auto.grab(2)
print("grabbed:", lm.auto.settings["grabbed"], " label", lm.auto.settings["grab_label"])
lm.auto.clear_grab()
print("after clear_grab:", lm.auto.settings["grabbed"])

# %% [markdown]
# **AUTO's buttons.** Everything in the AUTO window is in `m.auto.buttons`,
# e.g. its own SVG of the diagram and the all-info dump.

# %%
lm.auto.buttons.file.svg("diagram.svg")
print(lm.transcript.strip().splitlines()[-1])
import shutil
shutil.copy(lm.workdir / "diagram.svg", FIG / "xpp_auto_diagram.svg")
lm.auto.buttons.file.all_info("allinfo.dat")
print((lm.workdir / "allinfo.dat").read_text().splitlines()[0][:100])
