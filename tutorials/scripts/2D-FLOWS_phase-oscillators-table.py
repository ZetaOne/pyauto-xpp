"""
Two phase oscillators with a tabulated coupling
===============================================

Two oscillators with phases x1, x2 and frequencies w1, w2 interact through a
function h that is given as a table of 101 samples on [0, 2*pi] (XPP's
`table h % 101 0 2pi expr`, linearly interpolated):
    x1' = w1 + a h(x2 - x1),   x2' = w2 + a h(x1 - x2).
The phase difference phi = x2 - x1 obeys a single equation,
    phi' = (w2 - w1) + a G(phi),   G(phi) = h(-phi) - h(phi),
so the pair locks (phi settles) exactly when |w2 - w1| <= a max|G|; outside
that band phi drifts, slowly near the edge.

This script
  1. builds the model with ModelBuilder (with the table) and prints the .ode file
  2. plots phi(t) inside and outside the locking band
  3. computes G from the tabulated h and, from it, the locking band, the locked
     phase and the drift (beat) frequency 2*pi / integral dphi / (w2 - w1 + a G)
  4. compares the beat frequency against w2 and the locked region in the
     (w2, a) plane with that theory
  5. checks the simulation against the theory

Run:   python 2D-FLOWS_phase-oscillators-table.py
Figures go to figures/2D-FLOWS_phase-oscillators-table/.

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
PER = 6.283185                            # the period used in the model (table range, mod, fold)
checks = []                               # (description, computed, expected, passed)


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
H_EXPR = "sin(t)+.6+.4*cos(t)+.2*sin(2*t)"
mb = (ModelBuilder("phase_oscillators_table")
      .comment("two phase oscillators coupled through a tabulated function h")
      .table("h", n=101, lo=0, hi=PER, expr=H_EXPR)
      .ode("x1", "w1+a*h(mod(x2-x1,6.283185))")
      .ode("x2", "w2+a*h(mod(x1-x2,6.283185))")
      .aux("phi", "mod(x2-x1,6.283185)")
      .parameter(w1=1, w2=1.1, a=0.15)
      .initial(x1=0, x2=2)
      .option(fold="x1", tor_per=PER)
      .option(fold="x2")
      .option(total=200, dt=0.05))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
w1 = 1.0


def drift(tr):
    """phi unwrapped, and the times at which it has gained each further full turn"""
    u = np.unwrap(tr["phi"], period=PER)
    k = np.floor((u - u[0]) / PER)
    i = np.where(np.diff(k) != 0)[0]
    t = tr.t
    tc = t[i] + (u[0] + PER * np.maximum(k[i], k[i + 1]) - u[i]) * (t[i + 1] - t[i]) / (u[i + 1] - u[i])
    return u, tc


def beat_frequency(w2, a, total=2000):
    _, tc = drift(m.run(keep=False, w2=w2, a=a, total=total, trans=200))
    return PER / np.diff(tc).mean() if len(tc) > 2 else 0.0


def is_locked(w2, a, total=2500):
    u, _ = drift(m.run(keep=False, w2=w2, a=a, total=total, trans=total / 5))
    return abs(u[-1] - u[0]) < 0.05


# %% 2. phi against time
fig, ax = plt.subplots(figsize=(7, 3.8))
for w2, col in ((1.1, "tab:blue"), (1.3, "tab:green"), (1.4, "tab:red")):
    tr = m.run(keep=False, w2=w2, total=300)
    p = tr["phi"].copy()
    p[1:][np.abs(np.diff(p)) > PER / 2] = np.nan              # break the line where phi wraps
    ax.plot(tr.t, p, color=col, lw=1.4, label=f"w2 = {w2}")
ax.set(xlim=(0, 300), ylim=(0, 7.5), xlabel="t", ylabel="phi = x2 - x1 (mod 2π)",
       title="Phase difference, a = 0.15 (w1 = 1)")
ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
ax.legend(fontsize=8, loc="upper center", ncol=3)
fig.tight_layout()
fig.savefig(FIG / "phase_difference.png", dpi=130)

# %% 3. theory from the tabulated h (linear interpolation of the same 101 samples XPP uses)
tk = np.linspace(0, PER, 101)
hk = np.sin(tk) + .6 + .4 * np.cos(tk) + .2 * np.sin(2 * tk)


def G(p):
    h = lambda u: np.interp(np.mod(u, PER), tk, hk)   # noqa: E731
    return h(-p) - h(p)


grid = np.linspace(0, PER, 400001)[:-1]
Gmax = np.abs(G(grid)).max()


def beat_theory(delta, a):
    """2*pi / (time for phi to make one turn); 0 inside the locking band"""
    if abs(delta) <= a * Gmax:
        return 0.0
    return PER / (np.mean(1 / np.abs(delta + a * G(grid))) * PER)


print(f"\nmax |G| from the table = {Gmax:.6f}: lock iff |w2 - w1| <= {Gmax:.4f} a")

# %% 4. beat frequency against w2, and the locked region in (w2, a)
W2 = np.linspace(0.4, 1.6, 61)
sim = [beat_frequency(w, 0.15, total=1000) for w in W2]
fine = np.linspace(0.4, 1.6, 1201)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
a1.plot(fine, [beat_theory(w - w1, 0.15) for w in fine], color="tab:orange", lw=2, label="theory from G")
a1.plot(W2, sim, "o", color="tab:blue", ms=4, label="simulation")
a1.set(xlabel="w2", ylabel="beat frequency of phi", title="Locking band at a = 0.15 (w1 = 1)")
a1.legend(fontsize=8)
A, W2g = np.linspace(0, 0.3, 21), np.linspace(0.4, 1.6, 49)
L = np.array([[is_locked(w, aa, total=800) for w in W2g] for aa in A])
a2.pcolormesh(W2g, A, L, cmap="Blues", vmin=0, vmax=1.6, shading="nearest")
a2.plot(w1 + Gmax * A, A, "--", color="tab:orange", lw=2, label="|w2 - w1| = a max|G|")
a2.plot(w1 - Gmax * A, A, "--", color="tab:orange", lw=2)
a2.plot([], [], "s", color=plt.get_cmap("Blues")(0.62), label="locked (simulation)")
a2.set(xlim=(0.4, 1.6), ylim=(0, 0.3), xlabel="w2", ylabel="coupling a", title="Locked region")
a2.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "locking.png", dpi=130)


# %% 5. checks
def edge(a, side):
    lo, hi = w1, w1 + side * 2 * a * Gmax                   # locked at lo, drifting at hi
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if is_locked(mid, a) else (lo, mid)
    return (lo + hi) / 2


print("\nchecks:")
# the locked phase solves 0.1 + 0.15 G(phi) = 0 on the stable branch (G' < 0), near phi = 0.24
lo, hi = 0.0, 1.0
for _ in range(60):
    lo, hi = ((lo + hi) / 2, hi) if 0.1 + 0.15 * G((lo + hi) / 2) > 0 else (lo, (lo + hi) / 2)
check("locked phi at w2 = 1.1, a = 0.15 vs root of G", m.run(keep=False, total=400)["phi"][-1], lo, 1e-6)
check("upper edge of the band at a = 0.15", edge(0.15, +1), w1 + 0.15 * Gmax, 1e-4)
check("lower edge of the band at a = 0.15", edge(0.15, -1), w1 - 0.15 * Gmax, 1e-4)
check("beat frequency at w2 = 1.4, a = 0.15", beat_frequency(1.4, 0.15), beat_theory(0.4, 0.15), 1e-5)
# smooth h gives G = -2 sin(phi) - 0.4 sin(2 phi), whose max |G| is at cos(phi) = (-2 + sqrt(9.12))/3.2;
# linear interpolation on steps of 2pi/100 changes it by at most step^2/8 * max|G''| ~ 1.8e-3
cphi = (-2 + np.sqrt(9.12)) / 3.2
sphi = np.sqrt(1 - cphi**2)
check("max |G| of the table vs the smooth h", Gmax, 2 * sphi + 0.8 * sphi * cphi, (PER / 100) ** 2 / 8 * 3.6)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
