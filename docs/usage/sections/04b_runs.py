# %% [markdown]
# ## Run conditions as data: `Run`
#
# `m.run(total=..., iapp=...)` changes the model and the change stays. A
# `Run` keeps *how to integrate* apart from the model: numerics, the method
# and its settings, parameters, initial conditions, any other `@` option.
# Applying a `Run` changes the model for that run only; afterwards it is as
# it was. Fields left out keep the model's own values.

# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyxpp
from pyxpp import Run

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)
m = pyxpp.load(PKG / "examples" / "lecar.ode")
before = m.snapshot()

base = Run(total=200, dt=0.05, meth="rk4", name="rest")
spiking = base.with_pars(iapp=0.1).with_ics(v=-0.3, w=0).replace(name="spiking")
stiff = spiking.replace(meth="cvode", tol=1e-10, atol=1e-10, name="spiking, cvode")
late = spiking.replace(trans=150, nout=4, name="spiking, last 50, every 4th step")
print(spiking)

# %% [markdown]
# `run_many` runs each one; every result carries the `Run` that made it.

# %%
trs = m.run_many([base, spiking, stiff, late])
for t in trs:
    print(f"{t.run.name:32s} rows {len(t):5d}   t {t.t[0]:6.1f} .. {t.t[-1]:6.1f}   max v {t['v'].max():+.4f}")
print("model unchanged:", m.snapshot() == before)
print("rk4 against cvode (1e-10), largest |dv|:", f"{np.abs(trs[1]['v'] - trs[2]['v']).max():.1e}")
fig, ax = plt.subplots(figsize=(8, 3))
for t in trs:
    ax.plot(t.t, t["v"], lw=1, label=t.run.name)
ax.legend(fontsize=7); ax.set_xlabel("t"); ax.set_ylabel("v")
fig.tight_layout(); fig.savefig(FIG / "runs.png", dpi=100)

# %% [markdown]
# **Variants.** `replace(...)` changes fields (and replaces `pars`, `ics`,
# `options` whole); `with_pars`, `with_ics`, `with_options` add to them.
# `start="last"` continues from the end of the previous run (Initialconds >
# Last). `Run.from_model(m)` captures the model's current settings.

# %%
a = m.run(spiking)
b = m.run(spiking.replace(start="last"))
print("continues where the last run ended:", np.allclose(b.array[0, 1:3], a.array[-1, 1:3]))
print(Run.from_model(m).replace(pars={}, ics={}))

# %% [markdown]
# **One-off settings without a `Run`.** `keep=False` makes keyword settings
# apply to one run only (they are sorted into parameters, initial
# conditions, numerics and other options, and the result gets the `Run`);
# `m.using(...)` does the same for a block of runs.

# %%
t = m.run(total=50, iapp=0.1, v=-0.2, meth="cvode", keep=False)
print(t.run, " unchanged:", m.snapshot() == before)
with m.using(stiff):
    first = m.run()
    more = m.run_last()
    print("inside:", m.numerics["meth"], m.pars["iapp"])
print("after:", m.numerics["meth"], m.pars["iapp"], " unchanged:", m.snapshot() == before)

# %% [markdown]
# **Saving runs.** A `Run` is saved as JSON and loaded back, so a set of
# runs can live next to the results it produced. (Adaptive methods store
# every step: XPP sets `nout` to 1 for cvode, dopri, stiff, gear, rosen23.)

# %%
p = stiff.save("runs/spiking_cvode.json")
print(p.read_text())
print("loaded back equal:", Run.load(p) == stiff)
