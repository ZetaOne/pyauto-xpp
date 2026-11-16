# %% [markdown]
# ## Phase plane: nullclines, equilibria, direction fields
#
# `nullclines()` asks XPP to compute the nullclines in a 2D view and returns
# them as arrays of (x, y) points, with NaN rows between separate pieces.
# The key is the variable whose right-hand side is zero.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)

m = pyxpp.load(PKG / "examples" / "lecar.ode")
m.pars["iapp"] = 0.05
nc = m.nullclines(x="v", y="w", xlo=-0.6, xhi=0.5, ylo=-0.1, yhi=0.6)
print({k: a.shape for k, a in nc.items()})

# %% [markdown]
# `view()` sets the 2D window on its own (Viewaxes > 2D); `mesh=` refines
# the grid XPP uses for the nullclines (nUmerics > Ncline ctrl).

# %%
m.view(x="v", y="w", xlo=-0.6, xhi=0.5, ylo=-0.1, yhi=0.6)
fine = m.nullclines(mesh=200)
print("points with the default mesh:", len(nc["V"]), "  with mesh=200:", len(fine["V"]))

# %% [markdown]
# **Equilibria.** `equilibrium(guess)` is Newton's method (the core of
# Sing pts > Go) from a guess (a dict, a sequence, or the current initial
# conditions). It returns the point, its eigenvalues and its stability.
# At iapp = 0.05 Morris-Lecar has three equilibria.

# %%
eqs = []
for guess in ({"v": -0.4, "w": 0.0}, {"v": -0.15, "w": 0.03}, {"v": 0.0, "w": 0.3}):
    e = m.equilibrium(guess)
    eqs.append(e)
    print(e, " eigenvalues:", np.round(e.eigenvalues, 4))

# %% [markdown]
# The same with the button (Sing pts > Go): XPP asks whether to print the
# eigenvalues, and puts the result in the log. Sing pts > Monte Carlo
# searches from many random guesses.

# %%
m.ics.update(v=-0.15, w=0.03)
n = len(m.log)
m.buttons.sing_pts.go(True, False)        # print eigenvalues: yes; draw strong sets: no
print(m.transcript)
print(m.log[n:].strip())
n = len(m.log)
m.buttons.sing_pts.monte_carlo(0, 0, 200, 1e-3, -0.6, 0.5, -0.1, 0.6)
print(m.log[n:].strip())

# %% [markdown]
# **Phase portrait** from these pieces, with trajectories from a few starts.

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(*fine["V"].T, "C1", label="v' = 0")
ax.plot(*fine["W"].T, "C2", label="w' = 0")
for v0 in (-0.5, -0.3, -0.1, 0.2, 0.4):
    tr = m.run(total=150, v=v0, w=0.0)
    ax.plot(tr["v"], tr["w"], "k", lw=0.6)
for e in eqs:
    ax.plot(e.x["V"], e.x["W"], "o", color="k" if e.stable else "w", mec="k", ms=7)
ax.set(xlim=(-0.6, 0.5), ylim=(-0.1, 0.6), xlabel="v", ylabel="w", title="Morris-Lecar, iapp = 0.05")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "phase_plane.png", dpi=110)

# %% [markdown]
# **Direction field and XPP's own picture.** Dir.field/flow draws arrows on
# a grid; Graphic stuff > SVG writes the window as XPP draws it.

# %%
m.buttons.dir_field.direct_field(12)
m.buttons.nullcline.new()
m.buttons.graphic_stuff.svg("pp.svg")
print("XPP wrote", (m.workdir / "pp.svg").stat().st_size, "bytes:", m.transcript.strip().splitlines()[-1])
import shutil
shutil.copy(m.workdir / "pp.svg", FIG / "xpp_phase_plane.svg")

# %% [markdown]
# **Equilibria along a parameter** without AUTO: Sing pts > Range follows
# the equilibrium found from the current initial data as a parameter
# changes, and stores the results in the data browser: column `t` holds the
# parameter, then the equilibrium.

# %%
m.ics.update(v=-0.4, w=0.0)
m.buttons.sing_pts.range({"Range over": "iapp", "Steps": 20, "Start": 0, "End": 0.06})
d = m.data
print(d)
print("iapp, v, w:")
print(np.round(d.array[::5, :3], 5))
