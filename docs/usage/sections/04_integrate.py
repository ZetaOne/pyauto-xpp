# %% [markdown]
# ## Integrating and reading the data
#
# `m.run()` is Initialconds > Go: it integrates from the current initial
# conditions with the current numerics and returns the data browser as a
# `Trajectory`. Keyword arguments go through `set()` first.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)

m = pyxpp.load(PKG / "examples" / "lecar.ode")
tr = m.run(iapp=0.1, total=200, dt=0.05)
print(tr)
print("rows:", len(tr), " columns:", tr.names)
print("t:", tr.t[:3], "...", tr.t[-1])
print("v (any case):", tr["v"][:3], tr["V"][:3])
print("as numpy:", tr.array.shape, tr.array.dtype)
print(tr.to_pandas().describe().loc[["min", "max"]].round(4))

# %% [markdown]
# Auxiliary quantities (`aux` lines) are columns too. The data are what XPP
# stored: single precision, every `nout`-th step, after `trans`.

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(tr.t, tr["v"], label="v")
ax[0].plot(tr.t, tr["ik"] / 10, label="ik / 10 (aux)")
ax[0].set_xlabel("t"); ax[0].legend()
ax[1].plot(tr["v"], tr["w"]); ax[1].set(xlabel="v", ylabel="w")
fig.tight_layout(); fig.savefig(FIG / "lecar_run.png", dpi=110)

# %% [markdown]
# **Carrying on.**
#
# - `run_last()`: Initialconds > Last, start again from the end of the last
#   run (time starts again at `t0`);
# - Initialconds > Shift: the same, but time carries on (it sets `t0`);
# - `continue_run(T)`: Continue, extend the current run until time T (the
#   data then hold the whole run);
# - `last_point()`: the final state as a dict.

# %%
a = m.run(total=50)
b = m.run_last(total=50)
print("run_last starts at the end of the last run:", np.allclose(b.array[0, 1:3], a.array[-1, 1:3]),
      " (b starts at t =", b.t[0], ")")
m.buttons.initialconds.shift()
print("shift: t from", m.data.t[0], "to", m.data.t[-1], " t0 is now", round(m.numerics["t0"], 6))
m.set(t0=0)
c = m.continue_run(120)
print("continue_run: t from", c.t[0], "to", c.t[-1])
print("last_point:", m.last_point())

# %% [markdown]
# **Transient, output step and start time.** `trans` drops the first part,
# `nout` stores every n-th step, `t0` sets the starting time.

# %%
tr = m.run(total=100, trans=80, nout=4, t0=0)
print("first stored t:", tr.t[0], " step between rows:", tr.t[1] - tr.t[0], " rows:", len(tr))
m.set(trans=0, nout=1)

# %% [markdown]
# **Sweeps.** `sweep(name, values)` runs once per value of a parameter or an
# initial condition, and puts the old value back afterwards.

# %%
fig, ax = plt.subplots(figsize=(7, 3))
currents = [0.06, 0.08, 0.1, 0.2]
for I, run in zip(currents, m.sweep("iapp", currents, total=150)):
    ax.plot(run.t, run["v"], lw=1, label=f"iapp={I}")
ax.legend(fontsize=7); ax.set_xlabel("t")
fig.tight_layout(); fig.savefig(FIG / "sweep.png", dpi=110)
print("iapp after the sweep:", m.pars["iapp"])
starts = m.sweep("v", [-0.4, 0.0, 0.3], total=30, iapp=0)
print("end v from three starts:", [round(float(r["v"][-1]), 4) for r in starts])

# %% [markdown]
# **XPP's own range integration.** Initialconds > Range (a dialog, answered
# with a dict) integrates over a range in one go. With "Reset storage" = N
# the runs are stored one after the other.

# %%
m.set(total=40, iapp=0.0)
m.buttons.initialconds.range({"Range over": "iapp", "Steps": 4, "Start": 0.05,
                              "End": 0.25, "Reset storage (Y/N)": "N"})
print(m.transcript)
d = m.data
print("stored rows:", len(d), "= 5 runs x", len(d) // 5)

# %% [markdown]
# **Other ways to start.** Initialconds > Periodic (XPP 8.1) gives exactly
# one period of a stable orbit (see "Analysis tools"); New asks for each
# initial value;
# Backward integrates backwards in time (here it stops when `ik` passes
# XPP's bound, as in the GUI).

# %%
m.set(total=20, iapp=0.0)
m.buttons.initialconds.new(-0.1, 0.02)
print("new ICs:", m.ics.as_dict())
back = m.buttons.initialconds.backward()
print("backward run: t from", m.data.t[0], "to", m.data.t[-1])

# %% [markdown]
# **Methods compared.** A stiff solver with tolerances (CVODE through its
# button, which asks for the tolerances) against fixed-step RK4.

# %%
m.set(total=200, dt=0.05, iapp=0.1, v=-0.3, w=0)
rk = m.run(meth="rk4")
m.buttons.numerics.method.cvode(1e-10, 1e-10)
cv = m.run()
print(m.numerics["meth"], " largest |v_rk4 - v_cvode| =", f"{np.abs(rk['v'] - cv['v']).max():.2e}")
