"""
FitzHugh-Nagumo with a random on/off input
==========================================

The input to a FitzHugh-Nagumo cell switches on and off at random.  s is a
two-state Markov variable (XPP's "markov"): it jumps from off (0) to on (1)
at rate kon and back at rate koff, so the input is i0 + di*s, a random
telegraph signal.  With di = 0.8 the cell oscillates while the input is on
and rests while it is off.

This script
  1. builds the model with ModelBuilder and prints the .ode file it makes
  2. shows the input and the voltage for a stretch of time
  3. checks the input statistics: the fraction of time on, and the dwell
     times in each state, exponential with means 1/koff (on) and 1/kon (off)
  4. explains the firing from the noise-free model: the rest state is stable
     at i = 0 and unstable at i = 0.8, inside the Hopf range of i
  5. checks that spikes start only while the input is on (or as the end of a
     spike already under way) and that the cell then fires with the period
     of the noise-free oscillation

Run:   python 27_NOISE_telegraph-fhn-markov.py
Figures go to figures/27_NOISE_telegraph-fhn-markov/.

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
    NAME = "27_NOISE_telegraph-fhn-markov"
FIG = HERE / "figures" / NAME
FIG.mkdir(parents=True, exist_ok=True)
checks = []                               # (description, computed, expected, passed)
SEED = 1                                  # XPP's random seed: same seed, same switching, same numbers


def check(what, value, expected, tol):
    ok = abs(value - expected) <= tol
    checks.append((what, value, expected, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: {value:.6g} (expected {expected:.6g})")


# %% 1. the model
kon, koff, i0, di, eps, a, b = 0.01, 0.005, 0.0, 0.8, 0.08, 0.7, 0.8
mb = (ModelBuilder("telegraph_fhn")
      .comment("FitzHugh-Nagumo driven by a random telegraph input")
      .markov("s", [["0", "kon"], ["koff", "0"]])    # row = from state, column = to state
      .ode("v", "v-v^3/3-w+i0+di*s")
      .ode("w", "eps*(v+a-b*w)")
      .aux("input", "i0+di*s")
      .parameter(i0=i0, di=di, kon=kon, koff=koff, eps=eps, a=a, b=b)
      .initial(v=-1.2, w=-0.62, s=0)
      .option(total=600, dt=0.05))
print(mb.render())
m = mb.load(HERE / "odes" / f"{NAME}.ode")          # writes the .ode file and loads it into XPP
m.buttons.numerics.stochast.new_seed(SEED)         # stocHast > New seed
m.set(jac_eps=1e-8)      # Jacobian step for eigenvalues (XPP 8.1 ignores "@ jac_eps" in the file)

# %% 2. one long run; the first stretch as a picture
TOTAL = 200000
tr = m.run(keep=False, total=TOTAL)
t, s, v = tr.t, tr["s"], tr["v"]
fig, axs = plt.subplots(2, 1, figsize=(7.5, 4.2), sharex=True, gridspec_kw={"height_ratios": [1, 2.5]})
n = int(1500 / 0.05)
axs[0].fill_between(t[:n], tr["input"][:n], step="post", color="tab:orange", alpha=0.6)
axs[0].set(ylabel="input", ylim=(-0.1, 1), title="Random telegraph input and the response")
axs[1].plot(t[:n], v[:n], lw=0.9)
axs[1].set(xlabel="t", ylabel="v", xlim=(0, 1500))
fig.tight_layout()
fig.savefig(FIG / "time_series.png", dpi=130)

# %% 3. statistics of the input
frac_on = s.mean()
jumps = np.flatnonzero(np.diff(s) != 0) + 1                 # indices where s changes
dwell, state = np.diff(t[jumps]), s[jumps[:-1]]            # complete stays only (ends dropped)
on, off = dwell[state == 1], dwell[state == 0]
print(f"\n{len(jumps)} switches; fraction of time on {frac_on:.4f};"
      f" mean stay on {on.mean():.1f} ({len(on)}), off {off.mean():.1f} ({len(off)})")

fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, d, k, lab, col in ((axs[0], on, koff, "on", "tab:orange"), (axs[1], off, kon, "off", "tab:blue")):
    ax.hist(d, bins=40, range=(0, 5 / k), density=True, color=col, alpha=0.6, label=f"{len(d)} stays")
    x = np.linspace(0, 5 / k, 200)
    ax.plot(x, k * np.exp(-k * x), "k-", lw=1.5, label=f"{k} exp(-{k} t)")
    ax.set(yscale="log", xlabel=f"time spent {lab}", ylabel="density", title=f"Stays {lab}: mean 1/{k}")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "dwell_times.png", dpi=130)

# %% 4. the noise-free model: rest at i = 0, oscillation at i = 0.8
# Hopf points of FitzHugh-Nagumo (exact): trace 1 - v^2 - eps*b = 0 at v = -/+ c, c = sqrt(1 - eps*b),
# and i = v^3/3 - v + (v + a)/b on the curve of equilibria
c = np.sqrt(1 - eps * b)
i_hopf = sorted(sg * (c**3 / 3 - c + c / b) + a / b for sg in (-1, 1))
print(f"\nHopf range of the input: {i_hopf[0]:.4f} < i < {i_hopf[1]:.4f}")
eqs = {}
for inp, guess in ((i0, (-1.2, -0.6)), (i0 + di, (-0.3, 0.5))):
    m.set(i0=inp, di=0)
    eqs[inp] = m.equilibrium({"v": guess[0], "w": guess[1]})
    print(f"  input {inp}: {eqs[inp]}, eigenvalues {np.round(eqs[inp].eigenvalues, 4)}")
det_run = m.run(keep=False, total=600)                    # constant input 0.8, no switching
up = det_run.t[1:][(det_run["v"][:-1] < 1) & (det_run["v"][1:] >= 1)]
T = (up[-1] - up[2]) / (len(up) - 3)                      # mean period, first spikes dropped
m.set(i0=i0, di=di)
print(f"  period of the oscillation at input 0.8: {T:.2f}")

# %% 5. spikes and the input
starts = np.flatnonzero((v[:-1] < 1) & (v[1:] >= 1)) + 1   # v rises through 1
last_on = np.maximum.accumulate(np.where(s == 1, np.arange(len(s)), 0))   # last index with s = 1
late = (s[starts] == 0) & (t[starts] - t[last_on[starts]] > T)
print(f"\n{len(starts)} spikes: {np.sum(s[starts] == 1)} start while on,"
      f" {np.sum(s[starts] == 0)} just after the input went off, {late.sum()} later than {T:.1f} after")
# intervals between spikes in the same stay on (s = 1 throughout)
seg = np.cumsum(np.r_[0, np.diff(s) != 0])                 # stay number of every sample
isi = [t[j] - t[i] for i, j in zip(starts[:-1], starts[1:])
       if s[i] == 1 and seg[i] == seg[j]]
print(f"median interval between spikes within a stay on: {np.median(isi):.2f} ({len(isi)} intervals)")

# %% 6. checks
print("\nchecks:")
# a time average of a telegraph signal has variance 2 p (1-p) tau / T with tau = 1/(kon+koff):
# allow 4 standard errors
p = kon / (kon + koff)
check("fraction of time on kon/(kon+koff)", frac_on, p, 4 * np.sqrt(2 * p * (1 - p) / (kon + koff) / TOTAL))
# the mean of N exponential stays has standard error mean/sqrt(N): allow 4 of them
check("mean stay on 1/koff", on.mean(), 1 / koff, 4 / koff / np.sqrt(len(on)))
check("mean stay off 1/kon", off.mean(), 1 / kon, 4 / kon / np.sqrt(len(off)))
# for an exponential the standard deviation equals the mean (s.e. of the ratio ~ 1/sqrt(N))
check("stays on: std/mean = 1 (exponential)", on.std() / on.mean(), 1, 4 / np.sqrt(len(on)))
check("rest state stable at input 0 (1 = yes)", float(eqs[i0].stable), 1, 0)
check("rest state unstable at input 0.8 (1 = yes)", float(not eqs[i0 + di].stable), 1, 0)
check("input 0.8 inside the Hopf range (1 = yes)", float(i_hopf[0] < i0 + di < i_hopf[1]), 1, 0)
check("input 0 below the Hopf range (1 = yes)", float(i0 < i_hopf[0]), 1, 0)
# exact: the sum of the eigenvalues is the trace 1 - v^2 - eps*b at the equilibrium
v8 = eqs[i0 + di].x["V"]
check("trace at input 0.8 = 1 - v^2 - eps*b (exact)", float(np.sum(eqs[i0 + di].eigenvalues).real),
      1 - v8**2 - eps * b, 1e-5)
check("spikes starting later than one period after the input went off", late.sum(), 0, 0)
# spike times are resolved to the step dt = 0.05, so an interval is good to about 2 dt
check("interval between spikes while on = noise-free period", np.median(isi), T, 0.1)
n_ok = sum(c[3] for c in checks)
print(f"\n{NAME}: {n_ok}/{len(checks)} checks passed; figures in {FIG}")

# %% show the figures (VS Code: a window per figure; close them to finish)
plt.show()
