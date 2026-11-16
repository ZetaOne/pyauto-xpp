"""
Noisy FitzHugh-Nagumo: coherence resonance
==========================================

FitzHugh-Nagumo in its excitable regime, with white noise added to the slow
variable:  v' = (v - v^3/3 - u)/eps,  u' = v + a + sig*w,  where w is XPP's
"wiener" noise (integrated by Euler-Maruyama).  With a = 1.05 > 1 the rest state
is stable and nothing happens without noise.  Noise alone makes the cell
fire, and at a moderate noise level the spikes are at their most regular:
coherence resonance (Pikovsky and Kurths, 1997).

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. checks the rest state against the exact equilibrium and shows that
     without noise there are no spikes
  3. shows the spike trains and a noisy orbit in the phase plane
  4. sweeps the noise strength sig: spike rate, and the coefficient of
     variation (CV) of the inter-spike intervals, which has a minimum
  5. checks the results against exact values and the statistics of the runs

Run:   python NOISE_noisy-fhn-sde.py
Figures go to figures/NOISE_noisy-fhn-sde/.

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
    NAME = "NOISE_noisy-fhn-sde"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)
SEED = 1                                  # XPP's random seed: same seed, same noise, same numbers


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


def spike_times(tr):
    """Times where v rises through 1, counted again only after v has fallen below -1."""
    v, t = tr["v"], tr.t
    up = np.flatnonzero((v[:-1] < 1) & (v[1:] >= 1)) + 1        # rises through 1
    down = np.flatnonzero((v[:-1] > -1) & (v[1:] <= -1)) + 1    # falls through -1
    keep = [i for j, i in enumerate(up)                          # first rise after each fall
            if j == 0 or np.any((down > up[j - 1]) & (down < i))]
    return t[keep]


# %% 1. the model
mb = (ModelBuilder("noisy_fhn")
      .comment("FitzHugh-Nagumo with white noise on the recovery variable")
      .wiener("w")
      .ode("v", "(v-v^3/3-u)/eps")
      .ode("u", "v+a+sig*w")
      .parameter(eps=0.05, a=1.05, sig=0.2)
      .initial(v=-1.05, u=-0.664)
      .option(meth="euler", dt=0.002, total=60))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
m.buttons.numerics.stochast.new_seed(SEED)         # stocHast > New seed
# XPP's Jacobian (for eigenvalues) uses finite differences with step jac_eps, 1e-3 by
# default; that shifts the eigenvalues by about 1%.  XPP 8.1 ignores "@ jac_eps" in the
# .ode file (its defaults overwrite it), so set it here after loading.
m.set(jac_eps=1e-8)

# %% 2. no noise: a stable rest state, no spikes
eps, a = 0.05, 1.05
eq = m.equilibrium({"v": -1, "u": -0.7})
print("\nequilibrium:", eq, "\n  eigenvalues:", np.round(eq.eigenvalues, 5))
quiet = m.run(keep=False, sig=0, total=500, v=-0.9)       # start a little off rest
n_quiet = len(spike_times(quiet))
print(f"spikes in 500 time units without noise: {n_quiet}")

# %% 3. spike trains, and a noisy orbit in the phase plane
fig, axs = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True)
for ax, s in zip(axs, (0.04, 0.15, 0.8)):
    tr = m.run(keep=False, sig=s, total=60)
    ax.plot(tr.t, tr["v"], lw=0.8)
    ax.set(ylabel="v", ylim=(-2.4, 2.4))
    ax.set_title(f"sig = {s}", loc="right", fontsize=9)
axs[0].set_title("Spike trains driven by noise", loc="left")
axs[-1].set_xlabel("t")
fig.tight_layout()
fig.savefig(FIG / "spike_trains.png", dpi=130)

tr = m.run(keep=False, sig=0.15, total=40)
m.pars["sig"] = 0           # nullclines of the noise-free system (w holds the last noise sample)
nc = m.nullclines(x="v", y="u", xlo=-2.5, xhi=2.5, ylo=-1, yhi=1.65, mesh=200)
m.pars["sig"] = 0.2
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(tr["v"], tr["u"], lw=0.5, color="tab:blue", label="noisy orbit, sig = 0.15")
ax.plot(nc["V"][:, 0], nc["V"][:, 1], color="tab:orange", lw=2, label="v' = 0")
ax.plot(nc["U"][:, 0], nc["U"][:, 1], color="tab:green", lw=2, label="u' = 0 (no noise)")
ax.plot(eq.x["V"], eq.x["U"], "ko", ms=7, label="stable rest state")
ax.set(xlim=(-2.5, 2.5), ylim=(-1, 1.65), xlabel="v", ylabel="u", title="Noisy FitzHugh-Nagumo")
ax.legend(fontsize=8, loc="upper center", ncol=2)
fig.tight_layout()
fig.savefig(FIG / "phase_plane.png", dpi=130)

# %% 4. sweep the noise: rate and CV of the inter-spike intervals
TOTAL = 4000
sigs = np.array([0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.65, 1.0])
rate, rate_se, cv, cv_se = [], [], [], []
print(f"\n  sig   spikes    rate     CV")
for s in sigs:
    isi = np.diff(spike_times(m.run(keep=False, sig=s, total=TOTAL)))
    n, c = len(isi), isi.std() / isi.mean()
    rate.append(1 / isi.mean()); rate_se.append(c / isi.mean() / np.sqrt(n))
    cv.append(c); cv_se.append(c * np.sqrt((1 + 2 * c**2) / (2 * n)))   # large-sample s.e. of a CV
    print(f"  {s:4.2f}  {n + 1:6d}  {rate[-1]:.4f}  {c:.3f}")
rate, rate_se, cv, cv_se = map(np.array, (rate, rate_se, cv, cv_se))

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(sigs, cv, 2 * cv_se, fmt="o-", color="tab:blue", capsize=3, label="CV of intervals")
ax.set(xscale="log", xlabel="noise strength sig", ylabel="CV of inter-spike intervals")
ax.set_title("Coherence resonance (bars: 2 standard errors)")
ax2 = ax.twinx()
ax2.errorbar(sigs, rate, 2 * rate_se, fmt="s--", color="tab:red", capsize=3, ms=4)
ax2.set_ylabel("spike rate", color="tab:red")
ax.plot([], [], "s--", color="tab:red", label="spike rate")
ax.legend(fontsize=8, loc="upper center")
fig.tight_layout()
fig.savefig(FIG / "coherence_resonance.png", dpi=130)

# %% 5. checks
print("\nchecks:")
# exact: v' = 0 and u' = 0 give v = -a and u = v - v^3/3 = -a + a^3/3
check("rest state v = -a (exact)", eq.x["V"], -a, 1e-6)
check("rest state u = -a + a^3/3 (exact)", eq.x["U"], -a + a**3 / 3, 1e-6)
# Jacobian [[(1-v^2)/eps, -1/eps], [1, 0]]: trace (1-a^2)/eps = -2.05 < 0, det 1/eps > 0: stable
check("sum of eigenvalues (1-a^2)/eps (exact)", float(np.sum(eq.eigenvalues).real), (1 - a**2) / eps, 1e-4)
check("product of eigenvalues 1/eps (exact)", float(np.prod(eq.eigenvalues).real), 1 / eps, 1e-3)
check("spikes without noise", n_quiet, 0, 0)
# coherence resonance: the smallest CV is inside the sweep, below both ends by more than
# 4 combined standard errors (the intervals of a noise-driven train are nearly independent)
k = int(np.argmin(cv))
gap = min(cv[0] - cv[k] - 4 * np.hypot(cv_se[0], cv_se[k]),
          cv[-1] - cv[k] - 4 * np.hypot(cv_se[-1], cv_se[k]))
print(f"  CV is smallest at sig = {sigs[k]}")
check("CV minimum inside, clear of both ends (1 = yes)", float(0 < k < len(sigs) - 1 and gap > 0), 1, 0)
# the rate never falls by more than 4 standard errors from one sig to the next, and rises overall
drops = (rate[:-1] - rate[1:]) / np.hypot(rate_se[:-1], rate_se[1:])
check("rate never drops significantly (1 = yes)", float(drops.max() < 4), 1, 0)
check("rate rises from smallest to largest sig, in s.e. (> 4)",
      float((rate[-1] - rate[0]) / np.hypot(rate_se[0], rate_se[-1]) > 4), 1, 0)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
