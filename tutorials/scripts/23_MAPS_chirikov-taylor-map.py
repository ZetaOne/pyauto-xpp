"""
Chirikov-Taylor (standard) map: invariant curves and chaos
==========================================================

p -> p + K sin q,  q -> q + p (the new p), both taken modulo 2 pi: a kicked
rotor seen once per kick.  The map preserves area.  For K = 0 every horizontal
line p = const is invariant; as K grows these curves break up, islands and
chaotic layers appear, and above K = 0.9716 (Greene 1979) the last curve that
spans the whole q range is gone, so orbits can wander through all of p.

XPP Studio's file writes the kick as K sin(q)/(2 pi), so its K = 6 is K = 0.955
in the usual form used here.

This script
  1. builds the map with ModelBuilder (folded onto the torus) and prints the .ode
  2. draws phase portraits from many starts at K = 0.5, 0.9716, 2 and 6
  3. finds the fixed points (0, 0) and (pi, 0) with XPP and their multipliers:
     (0, 0) is a saddle, (pi, 0) is elliptic up to K = 4
  4. shows that the map preserves area, by mapping a small circle of starts
  5. checks the results against exact values

Run:   python 23_MAPS_chirikov-taylor-map.py
Figures go to figures/23_MAPS_chirikov-taylor-map/.

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
    NAME = "23_MAPS_chirikov-taylor-map"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
TWO_PI = 6.28318530718
mb = (ModelBuilder("chirikov_taylor_map")
      .comment("Chirikov-Taylor standard map on the torus")
      .fixed("temp", "p+K*sin(q)")
      .map("p", "temp")
      .map("q", "q+temp")
      .parameter(K=0.9716)
      .initial(p=4.1, q=1.2)
      .option(total=3000, meth="discrete", bound=1000)
      .option(fold="p", tor_per=TWO_PI)          # keep p and q in [0, 2 pi)
      .option(fold="q"))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP

# %% 2. phase portraits
starts = [(q0, p0) for p0 in np.linspace(0.05, TWO_PI - 0.05, 16) for q0 in (0.3, np.pi)]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for ax, K in zip(axs.flat, (0.5, 0.9716, 2.0, 6.0)):
    for k, (q0, p0) in enumerate(starts):
        tr = m.run(keep=False, K=K, q=q0, p=p0, total=1500)
        ax.plot(tr["q"], tr["p"], ".", ms=0.6, color=plt.cm.tab20(k % 20))
    ax.set(xlim=(0, TWO_PI), ylim=(0, TWO_PI), xlabel="q", ylabel="p", title=f"K = {K}")
    ax.set_aspect("equal")
fig.suptitle("Standard map: 32 orbits of 1500 iterates each")
fig.tight_layout()
fig.savefig(FIG / "phase_portraits.png", dpi=130)

# transport in p: does one orbit from the chaotic layer near p = 0 reach every p?
def p_coverage(K):
    p = m.run(keep=False, K=K, q=0.01, p=0.01, total=100000)["p"]
    return np.mean(np.histogram(p, bins=60, range=(0, TWO_PI))[0] > 0)


cov = {K: p_coverage(K) for K in (0.5, 2.0)}
print(f"\nshare of p in [0, 2pi) visited by one orbit from (0.01, 0.01): {cov}")

# %% 3. fixed points and multipliers
# exact: a fixed point needs p' = p (mod 2 pi) and q + p' = q (mod 2 pi): p = 0, sin q = 0.
# The Jacobian in (p, q) is [[1, K cos q], [1, 1 + K cos q]], determinant 1 (area preserving),
# trace 2 + K cos q.  At (0, 0) the trace is 2 + K > 2: a saddle with
# multiplier 1 + K/2 + sqrt((1 + K/2)^2 - 1).  At (pi, 0) it is 2 - K: for 0 < K < 4 the
# multipliers are exp(+/- i theta) with cos theta = 1 - K/2 (elliptic), and at K = 4 one is -1.
# XPP's Jacobian uses steps JAC_EPS * max(JAC_EPS, |x|); at p = 0 that is JAC_EPS^2, so
# JAC_EPS = 1e-4 (steps 1e-8 in p, 3e-4 in q) balances rounding and truncation errors.
m.set(newt_tol=1e-6, jac_eps=1e-4)
fp = {}
for K in (2.0, 3.9, 4.1):
    m.set(K=K)
    fp[K, "pi"] = m.equilibrium({"q": 3.0, "p": 0.05})
    fp[K, "0"] = m.equilibrium({"q": 0.05, "p": 0.05})
    print(f"K = {K}: (pi, 0) multipliers {np.round(fp[K, 'pi'].eigenvalues, 6)};"
          f" (0, 0) multipliers {np.round(fp[K, '0'].eigenvalues.real, 6)}")
m.set(K=0.9716)

# %% 4. area preservation: map a circle of 2000 starts and compare the enclosed areas
def shoelace(x, y):
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


th = np.linspace(0, 2 * np.pi, 2001)[:-1]
c, R = np.array([2.0, 3.0]), 0.2
Q0, P0 = c[0] + R * np.cos(th), c[1] + R * np.sin(th)
ratios = {}
fig, ax = plt.subplots(figsize=(5.5, 6.2))
ax.fill(Q0, P0, color="tab:gray", alpha=0.6, label="start")
for (K, n), col in zip([(0.9716, 3), (6.0, 1)], ("tab:blue", "tab:red")):
    img = np.array([[r["q"][-1], r["p"][-1]] for r in
                    (m.run(keep=False, K=K, q=q0, p=p0, total=n) for q0, p0 in zip(Q0, P0))])
    ctr = m.run(keep=False, K=K, q=c[0], p=c[1], total=n)
    ctr = np.array([ctr["q"][-1], ctr["p"][-1]])
    d = (img - ctr + np.pi) % TWO_PI - np.pi          # undo the folding around the image of the centre
    ratios[K, n] = shoelace(d[:, 0], d[:, 1]) / shoelace(Q0 - c[0], P0 - c[1])
    for sq in (-TWO_PI, 0, TWO_PI):                     # draw the copies across the fold too
        ax.fill(ctr[0] + d[:, 0] + sq, ctr[1] + d[:, 1], color=col, alpha=0.6,
                label=f"K = {K}, {n} iterate{'s' * (n > 1)}: area ratio {ratios[K, n]:.6f}" if sq == 0 else None)
ax.set(xlim=(0, TWO_PI), ylim=(0, TWO_PI), xlabel="q", ylabel="p", title="The map preserves area")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(FIG / "area.png", dpi=130, bbox_inches="tight")

# %% 5. checks
print("\nchecks:")
check("fixed point q = pi (exact)", fp[2.0, "pi"].x["Q"], np.pi, 1e-9)
check("K = 2: (pi, 0) multipliers sum to 2 - K = 0 (exact)", fp[2.0, "pi"].eigenvalues.sum().real, 0, 1e-6)
check("K = 2: |multiplier| at (pi, 0) = 1 (elliptic, exact)", abs(fp[2.0, "pi"].eigenvalues[0]), 1, 1e-6)
check("K = 2: saddle multiplier at (0, 0), 2 + sqrt(3) (exact)", fp[2.0, "0"].eigenvalues.real.max(),
      2 + np.sqrt(3), 1e-6)
check("K = 3.9: |multiplier| at (pi, 0) still 1 (exact)", abs(fp[3.9, "pi"].eigenvalues[0]), 1, 1e-6)
check("K = 4.1: (pi, 0) a flip saddle, -1.05 - sqrt(0.1025) (exact)",
      fp[4.1, "pi"].eigenvalues.real.min(), -1.05 - np.sqrt(0.1025), 1e-6)
# the polygon through 2000 points misses O((2 pi/2000)^2) ~ 1e-5 of the area of a smooth curve
check("area ratio after 3 iterates, K = 0.9716 (exact 1)", ratios[0.9716, 3], 1, 1e-5)
check("area ratio after 1 iterate, K = 6 (exact 1)", ratios[6.0, 1], 1, 1e-5)
check("K = 2 (> 0.9716): one orbit visits all p (share 1)", cov[2.0], 1, 0)
check("K = 0.5 (< 0.9716): invariant curves block it (1 = blocked)", float(cov[0.5] < 1), 1, 0)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
