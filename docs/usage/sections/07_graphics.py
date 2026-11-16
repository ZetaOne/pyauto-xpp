# %% [markdown]
# ## XPP's own graphics, and checking them against the data
#
# XPP still draws everything (into an invisible window). Graphic stuff >
# SVG and Postscript write what it drew, so a figure can come from XPP
# itself; `pyxpp.plot` has helpers that make the matching matplotlib figure
# and check that XPP's curve and the data agree to the pixel.

# %%
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)
from pyxpp import plot as xplot

m = pyxpp.load(PKG / "examples" / "lecar.ode")
tr = m.run(iapp=0.1, total=150)
m.buttons.xi_vs_t("v")                              # Xi vs t: plot v against t
m.buttons.window.fit()                              # Window > Fit
m.buttons.graphic_stuff.svg("v_vs_t.svg")
m.buttons.graphic_stuff.postscript({"BW-0/Color-1": 1}, "v_vs_t.ps")
print(sorted(p.name for p in m.workdir.glob("v_vs_t.*")))
shutil.copy(m.workdir / "v_vs_t.svg", FIG / "xpp_v_vs_t.svg")

# %% [markdown]
# **More curves in one window.** Graphic stuff > Add curve adds a curve (any
# columns, colour and line type); Freeze keeps the current curve when the
# next run is drawn; Export data writes the numbers of the curves on the
# screen.

# %%
m.buttons.graphic_stuff.add_curve({"X-axis": "t", "Y-axis": "w", "Color": 2})
m.buttons.window.window({"Y Lo": -0.45, "Y Hi": 0.5})
m.buttons.graphic_stuff.export_data("curves.dat")
print((m.workdir / "curves.dat").read_text().splitlines()[:3])
m.buttons.graphic_stuff.svg("two_curves.svg")
shutil.copy(m.workdir / "two_curves.svg", FIG / "xpp_two_curves.svg")
m.buttons.graphic_stuff.remove_all()

# %% [markdown]
# **3D views.** Viewaxes > 3D takes the axes, their ranges, and the 2D
# window the projected box is drawn in (`XLo`..`YHi`, about -2..2 for the
# whole box), in one dialog; 3d-params sets the viewing angles.

# %%
r = pyxpp.load(text="""
x'=-y-z
y'=x+a*y
z'=b+z*(x-c)
par a=0.2,b=0.2,c=5.7
init x=1,y=1,z=0
@ total=200,dt=0.02,trans=100
done
""")
rt = r.run()
r.buttons.viewaxes.three_d({"X-axis": "x", "Y-axis": "y", "Z-axis": "z", "Xmin": -12, "Xmax": 12,
                            "Ymin": -12, "Ymax": 12, "Zmin": 0, "Zmax": 25,
                            "XLo": -2, "XHi": 2, "YLo": -2, "YHi": 2})
r.press("r")                                        # Restore: redraw
r.buttons.graphic_stuff.svg("roessler3d.svg")
shutil.copy(r.workdir / "roessler3d.svg", FIG / "xpp_roessler3d.svg")
print(r.transcript.strip().splitlines()[-1])

# %% [markdown]
# **The helpers in `pyxpp.plot`.** Each makes a matplotlib figure from XPP's
# data; the `xpp_*` ones have XPP draw the same view as SVG; `compare_*`
# fit XPP's drawn curve to the data and report the largest distance in
# pixels (rounding to whole pixels alone gives up to about 0.5-1 px).
#
# | helper | does |
# |---|---|
# | `window(tr, var)` | t range and y range with a margin |
# | `plot_x_vs_t(tr, var, lim)` | one variable against t |
# | `plot_all_vs_t(tr, names)` | a panel per variable |
# | `plot_projection(tr, x, y)` | y against x |
# | `plot_3d(tr, x, y, z)` | 3D trajectory |
# | `xpp_graph(m, var, lim, path)` | XPP's own var-vs-t SVG |
# | `xpp_graph2d(m, x, y, path)` / `xpp_graph3d(m, x, y, z, path)` | XPP's own 2D / 3D SVG |
# | `compare_with_xpp_graph(tr, var, svg)` | pixel check for var vs t |
# | `compare_curve(tr, cols, svg)` | pixel check for any 2D / 3D view |

# %%
lim = xplot.window(tr, "v")
fig, ax = xplot.plot_x_vs_t(tr, "v", lim, title="v against t (matplotlib)")
fig.savefig(FIG / "mpl_v_vs_t.png", dpi=100)
svg = xplot.xpp_graph(m, "v", lim, "check_v.svg")
print("v vs t:", xplot.compare_with_xpp_graph(tr, "v", svg))

fig = xplot.plot_all_vs_t(rt, ["x", "y", "z"], title="Roessler")
fig.savefig(FIG / "mpl_all_vs_t.png", dpi=90)
fig, ax = xplot.plot_3d(rt, "x", "y", "z", title="Roessler")
fig.savefig(FIG / "mpl_3d.png", dpi=90)
svg2 = xplot.xpp_graph2d(r, "x", "y", "check_xy.svg", tr=rt)
svg3 = xplot.xpp_graph3d(r, "x", "y", "z", "check_xyz.svg", tr=rt)
c2 = xplot.compare_curve(rt, ["x", "y"], svg2)
c3 = xplot.compare_curve(rt, ["x", "y", "z"], svg3)
print("x-y view: max", round(c2["max_px"], 2), "px;  3D view: max", round(c3["max_px"], 2), "px")
