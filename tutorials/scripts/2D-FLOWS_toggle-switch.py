"""
Genetic toggle switch: bistability
==================================

Two genes repress each other (Gardner, Cantor & Collins 2000):
u' = a1/(1+v^beta) - u,  v' = a2/(1+u^gamma) - v.  With strong enough
repression there are two stable states (one gene on, the other off) separated
by a saddle; which one the cell ends in depends on where it starts.  Changing
one gene's synthesis rate a1 removes one of the stable states in a fold
(saddle-node).

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. draws the phase plane at a1 = a2 = 3: nullclines, the two stable nodes
     and the saddle with their eigenvalues, and trajectories
  3. colours a grid of starting points by the state they end in (the basins
     of attraction; their boundary is the saddle's stable manifold)
  4. follows the equilibria in a1 with AUTO and finds the two folds (LP)
  5. follows a fold in two parameters (a1, a2): the bistable wedge with its
     cusp
  6. checks the results against exact values

Run:   python 2D-FLOWS_toggle-switch.py
Figures go to figures/2D-FLOWS_toggle-switch/.

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
mb = (ModelBuilder("toggle_switch")
      .comment("Genetic toggle switch (Gardner, Cantor & Collins 2000)")
      .ode("u", "a1/(1+v^beta)-u")
      .ode("v", "a2/(1+u^gamma)-v")
      .parameter(a1=3, a2=3, beta=2, gamma=2)
      .initial(u=2.5, v=0.4)
      .option(total=30, dt=0.05))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
m.auto.parameters("a1", "a2", "beta", "gamma")      # AUTO may follow a1 and a2 later
# Sing pt ctrl: XPP's default Jacobian epsilon (1e-3) puts errors of about 1e-3 into the
# eigenvalues; a finite-difference step of 1e-8 and a tighter Newton tolerance fix that
m.buttons.numerics.sing_pt_ctrl(500, 1e-10, 1e-8)

# %% 2. phase plane at a1 = a2 = 3
win = (-0.2, 3.6, -0.2, 3.6)
nc = m.nullclines(x="u", y="v", xlo=win[0], xhi=win[1], ylo=win[2], yhi=win[3], mesh=200)
eqs = [m.equilibrium(g) for g in ({"u": 2.6, "v": 0.4}, {"u": 1.2, "v": 1.2}, {"u": 0.4, "v": 2.6})]
print("equilibria at a1 = a2 = 3:")
for e in eqs:
    print(f"  u = {e.x['U']:.5f}, v = {e.x['V']:.5f}, eigenvalues {np.round(e.eigenvalues.real, 5)}, "
          f"{'stable node' if e.stable else 'saddle'}")

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(nc["U"][:, 0], nc["U"][:, 1], color="tab:orange", lw=2, label="u' = 0")
ax.plot(nc["V"][:, 0], nc["V"][:, 1], color="tab:green", lw=2, label="v' = 0")
starts = [(0.2, 0.1), (3.4, 3.2), (0.1, 3.4), (3.4, 0.1), (2.0, 3.4), (0.8, 0.1), (1.0, 1.05)]
for u0, v0 in starts:
    tr = m.run(keep=False, u=u0, v=v0, total=30)
    col = "tab:red" if tr["u"][-1] > tr["v"][-1] else "tab:blue"
    ax.plot(tr["u"], tr["v"], color=col, lw=1.2)
    ax.plot(u0, v0, "k.", ms=6)
for e in eqs:
    ax.plot(e.x["U"], e.x["V"], "ko" if e.stable else "o", mfc="k" if e.stable else "white",
            mec="k", ms=9)
ax.plot([], [], "ko", label="stable node")
ax.plot([], [], "o", mfc="white", mec="k", label="saddle")
ax.set(xlim=win[:2], ylim=win[2:], xlabel="u", ylabel="v", title="Toggle switch, a1 = a2 = 3")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "phase_plane.png", dpi=130)

# %% 3. basins of attraction
g = np.linspace(0, 3.5, 36)
ends = np.array([[[m.run(keep=False, u=a, v=b, total=40)[x][-1] for x in "uv"] for a in g] for b in g])
diff = ends[..., 0] - ends[..., 1]          # > 0: ended with u on, < 0: v on, ~0: on the saddle
basin = np.select([diff > 0.5, diff < -0.5], [1, -1], 0)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.pcolormesh(g, g, basin, cmap=matplotlib.colors.ListedColormap(["#bfdbfe", "0.6", "#fecaca"]),
              vmin=-1.5, vmax=1.5, shading="nearest")
ax.plot(nc["U"][:, 0], nc["U"][:, 1], color="tab:orange", lw=1.5)
ax.plot(nc["V"][:, 0], nc["V"][:, 1], color="tab:green", lw=1.5)
for e in eqs:
    ax.plot(e.x["U"], e.x["V"], "o", mfc="k" if e.stable else "white", mec="k", ms=8)
ax.set(xlim=(0, 3.5), ylim=(0, 3.5), xlabel="u(0)", ylabel="v(0)",
       title="Basins: red ends with u on, blue with v on\n(grey: starts on u = v go to the saddle)")
fig.tight_layout()
fig.savefig(FIG / "basins.png", dpi=130)
n_u, n_v, n_s = (basin == 1).sum(), (basin == -1).sum(), (basin == 0).sum()
frac = n_u / (n_u + n_v)
print(f"\nstarts ending with u on: {n_u}, with v on: {n_v}, on the saddle (the diagonal u = v): {n_s}")

# %% 4. AUTO: equilibria in a1, with a2 = 3
m.run(total=100)                 # settle on the u-on state (u = 2.5, v = 0.4 start)
m.run_last(total=10)             # AUTO starts from the current state
up = m.auto.run("a1", start="steady", ds=0.01, dsmax=0.05, parmin=0.5, parmax=6, nmax=2000)
m.auto.reset()                   # a second run from scratch replaces the diagram: reset it first
m.set(a1=3)                      # AUTO leaves a1 at the end of the branch; put it back
m.run(total=100)
m.run_last(total=10)
eqd = m.auto.run("a1", start="steady", ds=-0.01, dsmax=0.05, parmin=0.5, parmax=6, nmax=2000)
lps = sorted(eqd.special("LP"), key=lambda p: p.x)
print("\nAUTO, folds in a1 (a2 = 3):", [round(p.x, 5) for p in lps])

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for d, lab in ((eqd, True), (up, False)):
    xs = np.array([p.x for p in d])
    us = np.array([p.u0["u"] for p in d])
    st = np.array([bool(p.stable) for p in d])
    ax.plot(xs, np.where(st, us, np.nan), "r-", lw=2.5, label="stable" if lab else None)
    ax.plot(xs, np.where(~st, us, np.nan), "k--", lw=1.2, label="unstable (saddle)" if lab else None)
ax.plot([p.x for p in lps], [p.u0["u"] for p in lps], "o", color="purple", ms=8, label="fold (LP)")
ax.axvspan(lps[0].x, lps[1].x, color="0.92", zorder=0)
ax.set(xlim=(0.5, 6), xlabel="a1", ylabel="u", title="Equilibria in a1 (a2 = 3): bistable between the folds")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "bifurcation.png", dpi=130)

# %% 5. AUTO: the fold in (a1, a2)
c1 = m.auto.run(grab=lps[0].label, param2="a2", ds=0.01, dsmax=0.05, parmin=0.5, parmax=6, nmax=2000)
c2 = m.auto.run(grab=lps[0].label, ds=-0.01)
fold_a1 = np.concatenate([[q.x for q in c] for c in (c1, c2)])
fold_a2 = np.concatenate([[q.par["a2"] for q in c] for c in (c1, c2)])
k = np.argmin(fold_a1 + fold_a2)
print(f"fold curve: {len(fold_a1)} points; cusp near (a1, a2) = ({fold_a1[k]:.4f}, {fold_a2[k]:.4f})")

fig, ax = plt.subplots(figsize=(5.5, 5))
for c in (c1, c2):
    ax.plot([q.x for q in c], [q.par["a2"] for q in c], color="tab:purple", lw=2)
ax.plot(fold_a1[k], fold_a2[k], "k*", ms=12, label="cusp")
ax.plot(3, 3, "rs", ms=7, label="a1 = a2 = 3 (bistable)")
ax.text(4.3, 4.0, "two stable states", ha="center")
ax.text(1.4, 4.8, "one state\n(v on)", ha="center")
ax.text(4.8, 1.3, "one state\n(u on)", ha="center")
ax.set(xlim=(0.5, 6), ylim=(0.5, 6), xlabel="a1", ylabel="a2", title="Folds in (a1, a2): the bistable wedge")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "two_parameter.png", dpi=130)

# %% 6. checks
print("\nchecks:")
# exact: at a1 = a2 = a the saddle is u = v = s with s^3 + s - a = 0
s = np.roots([1, 0, 1, -3]).real.max()                 # the one real root for a = 3
check("saddle u = v (root of s^3+s-3)", eqs[1].x["U"], s, 1e-6)
# Jacobian at u = v = s: [[-1, -2s^2/(1+s^2)], [same, -1]], eigenvalues -1 -/+ 2s^2/(1+s^2)
check("saddle unstable eigenvalue -1+2s^2/(1+s^2)", eqs[1].eigenvalues.real.max(),
      -1 + 2 * s**2 / (1 + s**2), 1e-5)
# exact: the asymmetric states of u = 3/(1+v^2), v = 3/(1+u^2): u + u*v^2 = v + v*u^2 gives
# u*v = 1 (when u != v), then u + v = 3, so u = (3 + sqrt 5)/2
check("stable node u = (3+sqrt5)/2", eqs[0].x["U"], (3 + np.sqrt(5)) / 2, 1e-6)
# exact: the model is symmetric under u <-> v when a1 = a2 and beta = gamma, and so is the grid
# exact: there u*v = 1, so the Jacobian's off-diagonal product is 4/(1+u^2)(1+v^2) = 4/9,
# and the eigenvalues are -1 -/+ 2/3
check("stable node eigenvalue -1/3 (exact)", eqs[0].eigenvalues.real.max(), -1 / 3, 1e-6)
check("basins are mirror images (fraction with u on)", frac, 0.5, 1e-12)
check("starts on the diagonal that reach the saddle", n_s, len(g), 0)
# exact fold curve: det J = 0 gives 4u^2v^2 = (1+u^2)(1+v^2); with a2 = v(1+u^2) this is
# a2 = 4v^3/(3v^2-1), so a2 = 3 means 4v^3 - 9v^2 + 3 = 0, and a1 = u(1+v^2), u^2 = (1+v^2)/(3v^2-1)
vr = np.array([r.real for r in np.roots([4, -9, 0, 3]) if abs(r.imag) < 1e-12 and 3 * r.real**2 > 1])
a1_exact = sorted(np.sqrt((1 + vr**2) / (3 * vr**2 - 1)) * (1 + vr**2))
check("first fold a1 (exact)", lps[0].x, a1_exact[0], 1e-4)
check("second fold a1 (exact)", lps[1].x, a1_exact[1], 1e-4)
# exact cusp: on u = v the fold condition 4u^4 = (1+u^2)^2 gives u = 1, so a1 = a2 = 2
check("cusp a1 (exact 2)", fold_a1[k], 2.0, 0.02)
check("cusp a2 (exact 2)", fold_a2[k], 2.0, 0.02)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
