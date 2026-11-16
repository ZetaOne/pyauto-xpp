"""
A piecewise affine map of the unit square
=========================================

(x, y) -> (y, frac(-x + a y)), from Ashwin and Goetz (SIADS 2006).  Away from
the lines where -x + a y crosses an integer the map is the linear map
M = [[0, 1], [-1, a]] plus an integer shift: determinant 1, so it preserves
area, and for |a| < 2 it is a rotation in disguise (eigenvalues exp(+/- i theta),
cos theta = a/2).  The cutting by frac() makes a rich mixture of periodic
islands and complicated, fractal sets of orbits.  XPP Studio's file uses
`floor`; XPP's own name for it is `flr`, used here.

This script
  1. builds the map with ModelBuilder and prints the .ode file it makes
  2. draws long orbits for several a
  3. checks that orbits stay in the unit square, that the map can be undone
     exactly, and that it preserves area
  4. finds the periods of many starts: every point has period 4 at a = 0 and
     period 6 at a = 1; at a = (sqrt 5 - 1)/2 = 2 cos(2 pi/5) the periodic islands
     have periods that are multiples of 5; a grid of starts is coloured by period
  5. checks the results against those exact facts

Run:   python MAPS_piecewise-affine-map.py
Figures go to figures/MAPS_piecewise-affine-map/.

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


# %% 1. the model
mb = (ModelBuilder("piecewise_affine_map")
      .comment("piecewise affine map on the unit square (Ashwin & Goetz 2006)")
      .parameter(a=0.5)
      .fixed("yy", "-x+a*y")
      .fixed("ny", "yy-flr(yy)")                  # fractional part
      .map("x", "y")
      .map("y", "ny")
      .initial(x=0.4319130445, y=0.2635796134)
      .option(maxstor=40000, meth="discrete", total=30000, bound=10000))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
GOLD = (np.sqrt(5) - 1) / 2                          # = 2 cos(2 pi/5)


def mod1_dist(u, v):
    """distance on the circle of length 1 (values just below 1 and just above 0 are close)"""
    d = np.abs(u - v) % 1
    return np.minimum(d, 1 - d)


# %% 2. long orbits
fig, axs = plt.subplots(2, 2, figsize=(9, 9))
in_square, inv_err = True, 0.0
for ax, a in zip(axs.flat, (0.3, 0.5, GOLD, 0.9)):
    tr = m.run(keep=False, a=a)                   # 30000 iterates from XPP Studio's start
    x, y = tr["x"], tr["y"]
    in_square &= bool(x.min() >= 0 and x.max() < 1 and y.min() >= 0 and y.max() < 1)
    # the inverse: y_n = x_{n+1}, and x_n = frac(a x_{n+1} - y_{n+1}) since x_n is in [0, 1)
    inv_err = max(inv_err, mod1_dist((a * x[1:] - y[1:]) % 1, x[:-1]).max())
    ax.plot(x, y, ",k")
    title = f"a = {a:.6g}"
    if a == GOLD:                                 # this start has period 10: add 40 more starts
        for k, (x0, y0) in enumerate(np.random.default_rng(0).random((40, 2))):
            o = m.run(keep=False, a=a, x=x0, y=y0, total=3000)
            ax.plot(o["x"], o["y"], ".", ms=0.5, color=plt.cm.tab20(k % 20))
        title += " (41 orbits)"
    ax.set(xlim=(0, 1), ylim=(0, 1), title=title, xlabel="x", ylabel="y")
    ax.set_aspect("equal")
fig.suptitle("Orbits of 30000 iterates from XPP Studio's start")
fig.tight_layout()
fig.savefig(FIG / "orbits.png", dpi=150)
print(f"\norbits stay in the unit square: {in_square}; largest error of the inverse map {inv_err:.2e}")

# %% 3. area: a square of starts, mapped once (away from the cutting lines the map is
# affine, so the image of the square is exactly a parallelogram of the same area)
def shoelace(x, y):
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


sq = np.array([[0.1, 0.5], [0.3, 0.5], [0.3, 0.7], [0.1, 0.7]])     # -x + a y stays in (0, 1)
img = np.array([[r["x"][-1], r["y"][-1]] for r in
                (m.run(keep=False, a=GOLD, x=x0, y=y0, total=1) for x0, y0 in sq)])
area_ratio = shoelace(*img.T) / shoelace(*sq.T)
print(f"area of a square after one iterate / before: {area_ratio:.8f}")

# %% 4. periods
rng = np.random.default_rng(1)


def periods(a, pts, n_iter):
    """smallest p <= n_iter with F^p(z) = z for each start z (0 if none found)"""
    out = []
    for x0, y0 in pts:
        tr = m.run(keep=False, a=a, x=x0, y=y0, total=n_iter)
        hit = np.where((mod1_dist(tr["x"][1:], tr["x"][0]) < 1e-5) &
                       (mod1_dist(tr["y"][1:], tr["y"][0]) < 1e-5))[0]
        out.append(hit[0] + 1 if len(hit) else 0)
    return np.array(out)


per0, per1 = periods(0.0, rng.random((200, 2)), 50), periods(1.0, rng.random((200, 2)), 50)
g = (np.arange(150) + 0.5) / 150                     # a 150 x 150 grid of starts
pts = np.array([(x0, y0) for y0 in g for x0 in g])
perg = periods(GOLD, pts, 1200)
found = perg[perg > 0]
print(f"a = 0: periods {sorted(int(p) for p in set(per0))}; a = 1: periods {sorted(int(p) for p in set(per1))}")
print(f"a = golden: periods found {sorted(int(p) for p in set(found))}; {np.mean(perg == 0):.1%} of starts not periodic within 1200")

fig, ax = plt.subplots(figsize=(6.4, 5.6))
img = np.where(perg > 0, np.log10(np.maximum(perg, 1)), np.nan).reshape(150, 150)
cmap = plt.cm.viridis.copy()
cmap.set_bad("tab:red")                               # red: no period up to 1200
sc = ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, interpolation="nearest")
ax.plot([], [], "s", color="tab:red", label="period > 1200")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("log10(period)")
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y",
       title="Periods of a grid of starts, a = (sqrt 5 - 1)/2")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "periods_golden.png", dpi=150)

# %% 5. checks
print("\nchecks:")
check("orbits stay in [0, 1)^2 (1 = yes)", float(in_square), 1, 0)
# XPP stores orbits in single precision (about 1e-7), hence the 1e-6 tolerances
check("the inverse map reproduces the orbit, largest error (exact 0)", inv_err, 0, 1e-6)
# each stored corner is off by up to 2^-24 per coordinate: the area by about perimeter * 2^-24,
# relative to an area of 0.04 that allows 0.8 * 6e-8 / 0.04 ~ 1.2e-6; tolerance 4 times that
check("area ratio of a square, det M = 1 (exact)", area_ratio, 1, 5e-6)
# a = 0: M = [[0, 1], [-1, 0]] is integer with M^4 = I, so F^4(z) = z + integer = z
check("a = 0: share of starts with period 4 (exact 1)", np.mean(per0 == 4), 1, 0)
# a = 1: M = [[0, 1], [-1, 1]] is integer with M^6 = I, so F^6 = identity
check("a = 1: share of starts with period 6 (exact 1)", np.mean(per1 == 6), 1, 0)
# a = 2 cos(2 pi/5): on a periodic island F^p is M^p plus a shift, which fixes an open set only
# if M^p = I, i.e. p is a multiple of 5
check("a = golden: share of periods that are multiples of 5 (exact 1)", np.mean(found % 5 == 0), 1, 0)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
