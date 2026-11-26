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

# %% [markdown]
# ### Maps: zero-change curves, stable manifolds, cobwebs (XPPAUT 8.4, 25 Nov)
#
# For a map $(x, y) \mapsto (f, g)$ (`meth=discrete`) XPP's Nullcline command
# draws the **zero-change curves** $f(x,y) = x$ and $g(x,y) = y$, which cross
# at the fixed points; `nullclines()` returns them unchanged.
# `stable_manifold(near=...)` computes the stable manifold of a saddle
# **without the inverse map** (England, Krauskopf & Osinga, SIAM J. Appl.
# Dyn. Syst. 3 (2004) 161-190), XPP's Sing pts > draw invariant sets for a
# map. Its settings are XPP's (Numerics > stabLe manifold, or `@ MFLEN=...`):
# `arclength`, `init`, `alpha`, `dalpha`, `delta`, and `box`, the region where
# the branches stop (by default the view enlarged by half its size on each side).

# %%
ikeda = pyxpp.load(text="""
par a=1,b=.9,e=1,phi=.4,q=6
m=phi-q/(1+x^2+y^2)
x'=a+b*x*cos(m)-e*y*sin(m)
y'=b*y*cos(m)+e*x*sin(m)
@ meth=discrete,total=200
@ xp=x,yp=y,xlo=-9,xhi=2,ylo=-5,yhi=3
done
""")
zc = ikeda.nullclines(mesh=200)
ws = ikeda.stable_manifold(near={"x": 1.08, "y": -2.41})
print("saddle", ws["saddle"], " stable multiplier", round(ws["multiplier"], 6))
for b, why, L in zip(ws["branches"], ws["why"], ws["arclength"]):
    print(f"  branch: {len(b)} points, arclength {L:.2f}, {why}")
fig, ax = plt.subplots(figsize=(6, 4.6))
ax.plot(*zc["X"].T, color="tab:red", lw=0.7, label="x does not change")
ax.plot(*zc["Y"].T, color="tab:green", lw=0.7, label="y does not change")
for b in ws["branches"]:
    ax.plot(b[:, 0], b[:, 1], color="tab:blue", lw=0.6)
ax.plot(ws["saddle"]["X"], ws["saddle"]["Y"], "s", mfc="w", mec="k")
ax.set(xlim=(-9, 2), ylim=(-5, 3), xlabel="x", ylabel="y", title="Ikeda-type map: W^s of a saddle (blue)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "map_stable_manifold.png", dpi=110)

# %% [markdown]
# **Cobwebs.** XPP 8.4 draws a cobweb for a one-dimensional map (Viewaxes >
# Cobweb). `pyxpp.plot.cobweb(m)` draws the same picture with matplotlib, and
# `cobweb_data(m)` returns its pieces: the graph of $f$, the staircase of the
# orbit, and the fixed points with their multipliers.

# %%
from pyxpp.plot import cobweb
logistic = pyxpp.load(text="par r=3.2\ninit x=0.2\nx'=r*x*(1-x)\n@ meth=discrete,total=200\ndone\n")
fig, ax = plt.subplots(figsize=(4.4, 4.4))
_, cw = cobweb(logistic, n=40, ax=ax, title="logistic map, r = 3.2")
print("fixed points (x*, multiplier):", [(round(x, 6), round(mu, 6)) for x, mu in cw["fixed"]])
fig.tight_layout(); fig.savefig(FIG / "cobweb.png", dpi=110)
