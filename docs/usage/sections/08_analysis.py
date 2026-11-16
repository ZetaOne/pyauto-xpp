# %% [markdown]
# ## Analysis tools in nUmerics
#
# These buttons replace XPP's data browser with their result (a histogram,
# a spectrum, an adjoint, ...). `m.data` always shows what the browser
# shows; nUmerics > stocHast > Data puts the solution back.
#
# **Statistics of a column, histogram, Fourier series.**

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)

m = pyxpp.load(PKG / "examples" / "lecar.ode")
S = m.buttons.numerics.stochast
tr = m.run(iapp=0.1, total=400, trans=200, dt=0.05)

S.stat("v")                                       # mean and standard deviation of a column
print(m.transcript.strip().splitlines()[-1], "  numpy:", round(float(tr["v"].mean()), 6), round(float(tr["v"].std()), 6))

S.histogram(50, -0.4, 0.3, "v", "")               # bins, low, high, variable, condition
h = m.data
print("histogram:", h, " columns t = bin, V = count; total count", int(h["v"].sum()))
S.data()                                          # back to the solution
S.fourier("v")
f = m.data
print("Fourier:", f, " t = frequency, V = cosine, W = sine coefficient")
k = 1 + np.argmax(np.hypot(f["v"][1:], f["w"][1:]))
print("strongest mode: frequency", round(float(f.t[k]), 5), "-> period", round(1 / float(f.t[k]), 3),
      "(frequency step 1/200: the run is 200 long)")
S.data()

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].bar(h.t, h["v"], width=0.014); ax[0].set(xlabel="v", ylabel="count", title="histogram of v")
ax[1].plot(f.t[:60], np.hypot(f["v"], f["w"])[:60], ".-"); ax[1].set(xlabel="frequency", title="|Fourier coefficient|")
fig.tight_layout(); fig.savefig(FIG / "hist_fourier.png", dpi=100)

# %% [markdown]
# **Largest Liapunov exponent** of the current orbit (about 0 on a limit
# cycle).

# %%
S.liapunov(0)                                     # 0: no range over parameters
print(m.transcript.strip().splitlines()[-1])

# %% [markdown]
# **Poincare map.** With a section set, XPP stores only the crossings
# (here v = 0 going up); the times between them are the period.

# %%
P = m.buttons.numerics.poincare_map
P.section({"Variable": "v", "Section": 0, "Direction (+1,-1,0)": 1})
cross = m.run(trans=0, total=300)
print(cross, " periods:", np.round(np.diff(cross.t)[-3:], 4))
P.none()

# %% [markdown]
# **One period of the orbit** (XPP 8.1: Initialconds > Periodic). After a
# run has settled on a stable orbit, Periodic finds the period in double
# precision, puts the initial condition at the peak of the plotted
# variable, sets `total` to the period and `trans` to 0, and integrates one
# period. This is the start that AUTO's periodic runs and the adjoint need.

# %%
m.run(iapp=0.1, total=300, dt=0.01, trans=0)
m.buttons.xi_vs_t("v")                              # Periodic starts at the peak of the plotted variable
m.buttons.initialconds.periodic()
print(m.transcript.strip().splitlines()[-1])
orbit = m.data
T = m.numerics["total"]
print(f"total = {T:.6f}, trans = {m.numerics['trans']}, v(0) = {orbit['v'][0]:.6f} = max v {orbit['v'].max():.6f}")
print("XPP's note:", m.log.strip().splitlines()[-1].strip())

# %% [markdown]
# **Adjoint and interaction function** (nUmerics > Averaging) on that one
# period. XPP 8.1 closes the orbit by Newton steps, normalises the adjoint,
# and also computes the isostable quantities. Make H asks for the coupling
# of each equation (`v'` is the other cell's v).

# %%
A = m.buttons.numerics.averaging
n = len(m.log)
A.new_adj()
adj = m.data
print(f"adjoint: {adj}, max of the V component {adj['v'].max():.4f}")
print("\n".join(l.strip() for l in m.log[n:].strip().splitlines()[-4:]))
A.make_h("v'-v", "0")                              # diffusive coupling through v
H = m.data
print("H:", H, " H(0) =", round(float(H["v"][0]), 5), "(0 for diffusive coupling: in phase, v' - v = 0)")

# %% [markdown]
# **Phase reduction to second order** (XPP 8.1, also in Averaging): the
# isostable response I(t), the Floquet exponent's function rho, the
# second-order phase correction B(t), and the second-order interaction
# function H2 of two coupled cells. Each replaces the browser's data; H2
# also reports gamma, computed two ways.

# %%
curves = {"adjoint Z (V)": (adj.t, adj["v"]), "H": (H.t, H["v"])}
for name, button in (("isostable I (V)", A.isostable), ("Floquet rho (V)", A.floquet_rho),
                     ("B correction (V)", A.b_correction)):
    button()
    d = m.data
    curves[name] = (d.t.copy(), d["v"].copy())
n = len(m.log)
A.h2()
h2 = m.data
curves["H2"] = (h2.t, h2["v"])
print("\n".join(l.strip() for l in m.log[n:].strip().splitlines()))
fig, axs = plt.subplots(2, 3, figsize=(12, 5.5))
for ax, (name, (x, y)) in zip(axs.flat, curves.items()):
    ax.plot(x, y); ax.axhline(0, color="0.8", lw=0.8); ax.set_title(name, fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "phase_reduction.png", dpi=100)
A.orbit()                                          # back to the orbit

# %% [markdown]
# **Boundary-value problems** (Bndryval): `bndry` lines give the conditions
# (`x` is the value at the start, `x'` at the end). Here x'' = -x with
# x(0) = 0, x(pi/2) = 1, whose solution is sin t: the shooting finds y(0) = 1.

# %%
b = pyxpp.load(text="""
x'=y
y'=-x
bndry x
bndry x'-1
init x=0,y=0.3
@ total=1.5707963,dt=0.01
done
""")
b.buttons.bndryval.show()
bt = b.data
print("y(0) =", round(float(bt["y"][0]), 6), "  x(pi/2) =", round(float(bt["x"][-1]), 6),
      "  largest |x - sin t| =", f"{np.abs(bt['x'] - np.sin(bt.t)).max():.1e}")

# %% [markdown]
# **Delay equations.** `delay(x, tau)` needs the maximum delay in the
# `delay` option. x' = -x(t - tau) is stable for tau < pi/2 and oscillates
# with growing amplitude beyond.

# %%
d = pyxpp.load(text="""
x'=-delay(x,tau)
par tau=1
init x=1
@ delay=5,total=60,dt=0.01
done
""")
fig, ax = plt.subplots(figsize=(7, 2.8))
for tau in (1.0, 1.5, 1.65):
    run = d.run(tau=tau)
    ax.plot(run.t, run["x"], label=f"tau = {tau}")
    print(f"tau = {tau}: |x| at the end {abs(float(run['x'][-1])):.3g}")
ax.legend(); ax.set_xlabel("t"); fig.tight_layout(); fig.savefig(FIG / "delay.png", dpi=100)

# %% [markdown]
# **Many noisy runs.** stocHast > Compute repeats a stochastic run (a range
# dialog: here 200 runs from the same start); Mean and Variance then show
# the average over the runs at each time. For dx = -x dt + 0.3 dW the
# variance tends to 0.045.

# %%
s = pyxpp.load(text="""
wiener w
x'=-x+0.3*w
init x=1
@ total=5,dt=0.01,meth=euler
done
""")
SS = s.buttons.numerics.stochast
SS.new_seed(12345)
SS.compute({"Range over": "x", "Steps": 200, "Start": 1, "End": 1})
SS.mean(); mean = s.data
SS.variance(); var = s.data
tt = mean.t
print("mean at t=1, 5:", round(float(mean["x"][100]), 4), round(float(mean["x"][-1]), 4), "  theory", round(np.exp(-1), 4), round(np.exp(-5), 4))
print("variance at t=5:", round(float(var["x"][-1]), 4), "  theory", round(0.045 * (1 - np.exp(-10)), 4))
