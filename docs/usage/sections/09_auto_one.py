# %% [markdown]
# ## AUTO I: one-parameter diagrams
#
# `m.auto` is XPP's own AUTO window (XPP's built-in AUTO, not
# AUTO-07p). A run returns the new points as a `Diagram`; `m.auto.diagram`
# is the whole diagram so far.
#
# **Start on an equilibrium.** AUTO starts from the current initial
# conditions, which must be at (or very near) an equilibrium: integrate to
# rest first, then `run(param, start="steady")`. Numerics can be given in
# the same call.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)

m = pyxpp.load(PKG / "examples" / "lecar.ode")
m.pars["iapp"] = 0.0
m.run(total=300)
m.run_last(total=10)                      # sit on the equilibrium
eq = m.auto.run("iapp", start="steady", ds=0.01, dsmax=0.02, parmin=-0.2, parmax=0.5, nmax=2000)
print(eq)
for p in eq.labeled():
    print(f"  label {p.label:2d} {p.type:2s} iapp = {p.x:9.5f}  v = {p.u0['v']:8.5f}  stable: {p.stable}")

# %% [markdown]
# **Periodic orbits from a Hopf point.** `grab` a labelled point and choose
# what to do there; at an HB point XPP offers periodic / extend / new point
# / two param.

# %%
hb = eq.special("HB")[0]
po = m.auto.run(grab=hb.label, choice="periodic", ntst=40, dsmax=0.05, nmax=400)
print(po)
lp = [p for p in po.special("LP")]
print("folds of periodic orbits at iapp =", [round(p.x, 5) for p in lp])
print("period at the last point:", round(po[-1].period, 1), " at iapp =", round(po[-1].x, 5))

# %% [markdown]
# The orbits fold at iapp = 0.242 (the Hopf point is subcritical) and end
# where their period becomes infinite, at the lower fold of equilibria
# (iapp = 0.0833): a saddle-node on the orbit (SNIC). AUTO stops there and
# marks the last point MX.
#
# **The Diagram and its Points.** A `Point` has the parameter values
# (`par`), the state (`u0`; for an orbit also `umax`, `umin`, `ubar`), the
# period, the norm, `multipliers` (what AUTO stores: Floquet multipliers on an
# orbit, exp(eigenvalue) at an equilibrium), `eigenvalues` (an equilibrium's
# own eigenvalues, computed by XPP at that point), `solution` (the whole orbit
# on AUTO's mesh, for labelled points), stability, the AUTO branch, point
# number, label and type. `par` holds
# AUTO's parameter list (at most 8; see `parameters()` below).

# %%
p = lp[0]
print(p.type, p.label, "branch", p.branch, "point", p.pt, " periodic:", p.periodic, " stable:", p.stable)
print("  par:", {k: round(v, 5) for k, v in p.par.items()})
print("  max v", round(p.umax["v"], 4), " min v", round(p.umin["v"], 4), " mean v", round(p.ubar["v"], 4))
print("  period", round(p.period, 4), " norm", round(p.norm, 4))
print("  multipliers:", np.round(p.multipliers, 4))
d = m.auto.diagram
print("whole diagram:", d, " labels", [q.label for q in d.labeled()][:6], "...")
print("label 2:", d.label(2).type, " branches:", [len(b) for b in d.branches()])

# %% [markdown]
# **Tables.** `table(y)` is a numpy structured array (parameter, max and min
# of `y`, stability, ...); `to_pandas(y)` the same as a DataFrame.

# %%
df = d.to_pandas("v")
print(df.head(4).to_string())
print(df[df.type != ""][["label", "type", "par", "ymax", "ymin", "period", "stable", "periodic"]].round(4).to_string())

# %% [markdown]
# **Plot.** `Diagram.plot(y)` draws it as XPP does: red stable / black
# unstable equilibria, filled / open circles for stable / unstable orbits
# (max and min of y), special points labelled.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
d.plot(y="v", ax=ax)
ax.set_title("Morris-Lecar: equilibria and periodic orbits in iapp")
fig.tight_layout(); fig.savefig(FIG / "lecar_diagram.png", dpi=110)

# %% [markdown]
# **Checked against simulation.** The stable orbit's max and min of v at
# iapp = 0.1 against a long run at that current.

# %%
rows = d.table("v")
stab = rows[(rows["periodic"]) & (rows["stable"])]
k = np.argmin(np.abs(stab["par"] - 0.1))
tr = m.run(iapp=stab["par"][k], total=500, trans=400, v=0, w=0.3)
print(f"iapp = {stab['par'][k]:.4f}: AUTO max/min v = {stab['ymax'][k]:.4f} / {stab['ymin'][k]:.4f};"
      f"  simulation {tr['v'].max():.4f} / {tr['v'].min():.4f}")
m.set(trans=0)

# %% [markdown]
# **Settings.** `settings` shows AUTO's current numerics and choices;
# `numerics(...)` changes them (names as in the AutoNum dialog: ntst, nmax,
# npr, ds, dsmin, dsmax, ncol, epsl, epsu, epss, parmin, parmax, normmin,
# normmax, iad, mxbf, iid, itmx, itnw, nwtn, iads); `parameters(...)` sets
# the list of up to 8 parameters AUTO may use; `axes(...)` the plot type
# (hi, norm, hi-lo, period, two_par, frequency, average), y variable and
# ranges.

# %%
a = m.auto
a.numerics(ntst=60, npr=50, epsl=1e-6)
a.parameters("iapp", "gca", "phi")
a.axes("period", y="v", xmin=0, xmax=0.3)
s = a.settings
print({k: s[k] for k in ("ntst", "nmax", "npr", "ds", "dsmax", "parmin", "parmax", "epsl", "param", "y", "plot")})
print("AUTO parameters:", s["parameters"][:3])
a.axes("hi")
