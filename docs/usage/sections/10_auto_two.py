# %% [markdown]
# ## AUTO II: two-parameter curves
#
# Grab a fold (LP) or Hopf (HB) point and continue it in two parameters.
# At an LP, XPP goes straight to a two-parameter run; at an HB choose
# `"two param"`. `param2=` names the second parameter. Run once with
# `ds > 0` and once with `ds < 0` to get both directions along the curve.
#
# **The cusp** x' = h + r x - x^3: folds lie on 4 r^3 = 27 h^2.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)

cusp = pyxpp.load(text="""
x'=h+r*x-x^3
par h=-1,r=1
init x=-1
@ total=50
done
""")
cusp.run(); cusp.run_last()
br = cusp.auto.run("h", start="steady", ds=0.01, dsmax=0.05, parmin=-1, parmax=1, nmax=300)
lp = br.special("LP")[0]
print(f"fold at h = {lp.x:.6f}  (exact 2/(3 sqrt 3) = {2 / (3 * np.sqrt(3)):.6f})")
up = cusp.auto.run(grab=lp.label, param2="r", ds=0.01, nmax=300, parmin=-1, parmax=1)
down = cusp.auto.run(grab=lp.label, ds=-0.01)
for c in (up, down):
    h = np.array([p.x for p in c]); r = np.array([p.par["r"] for p in c])
    print(f"{len(c)} points, two_par={c[0].two_par}, max |4r^3 - 27h^2| = {np.abs(4 * r**3 - 27 * h**2).max():.1e}")

# %% [markdown]
# **Morris-Lecar in (iapp, gca):** the fold curve of equilibria and the Hopf
# curve. `Diagram.plot(two_par=True)` draws the two-parameter points.

# %%
m = pyxpp.load(PKG / "examples" / "lecar.ode")
m.run(total=300); m.run_last(total=10)
eq = m.auto.run("iapp", start="steady", ds=0.01, dsmax=0.02, parmin=-0.2, parmax=0.5, nmax=2000)
fold = eq.special("LP")[0]
hopf = eq.special("HB")[0]
num = dict(nmax=400, dsmax=0.02, parmin=-0.2, parmax=1.5)
f1 = m.auto.run(grab=fold.label, param2="gca", ds=0.01, **num)
f2 = m.auto.run(grab=fold.label, ds=-0.01)
h1 = m.auto.run(grab=hopf.label, choice="two param", ds=0.01)
h2 = m.auto.run(grab=hopf.label, choice="two param", ds=-0.01)
print("fold curve:", len(f1) + len(f2), "points;  Hopf curve:", len(h1) + len(h2), "points")
print("special points on them:", sorted({(p.type, round(p.x, 3), round(p.par["gca"], 3)) for p in (*f1, *f2, *h1, *h2) if p.type not in ("", "EP", "MX")}))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
m.auto.diagram.plot(two_par=True, ax=ax[0])          # the whole diagram, as XPP draws it
ax[0].set(xlim=(-0.2, 0.6), ylim=(0.5, 1.5), title="Diagram.plot(two_par=True)")
ax = ax[1]                                           # the same curves, coloured by kind
for c, col, lab in ((f1, "C0", "folds (LP)"), (f2, "C0", None), (h1, "C3", "Hopf (HB)"), (h2, "C3", None)):
    ax.plot([p.x for p in c], [p.par["gca"] for p in c], color=col, label=lab)
ax.plot(fold.x, 1.0, "ko", ms=4); ax.plot(hopf.x, 1.0, "ks", ms=4); ax.legend()
ax.set(xlim=(-0.2, 0.6), ylim=(0.5, 1.5), xlabel="iapp", ylabel="gca", title="Morris-Lecar in (iapp, gca)")
fig.tight_layout(); fig.savefig(FIG / "lecar_two_par.png", dpi=110)

# %% [markdown]
# **Sub- or supercritical? GH points** (XPP 8.1, `hopf_l1.c`).
# `m.auto.lyapunov(point)` gives the first Lyapunov coefficient l1 and the
# frequency at a Hopf point: l1 < 0 supercritical (a stable orbit is born),
# l1 > 0 subcritical. In the XPP window a subcritical stretch of a Hopf
# curve is drawn purple and a sign change is marked GH; headless,
# `m.auto.gh_points(curve)` finds the sign changes the same way.

# %%
print("Hopf at gca = 1: l1 = %.3f, omega = %.4f" % m.auto.lyapunov(hopf))
ghs = m.auto.gh_points(h1) + m.auto.gh_points(h2)
for g in ghs:
    print(f"GH at iapp = {g['par']:.4f}, gca = {g['par2']:.4f}  (l1 {g['l1_before']:+.3f} -> {g['l1_after']:+.3f})")

# %% [markdown]
# **Checked:** between the two GH points the Hopf point should be
# supercritical, so the first orbits from it are stable (gca = 0.45 lies
# in that stretch of the curve, which crosses it twice); at gca = 1 it is
# subcritical, so they start unstable (and fold, as in section AUTO I).

# %%
for gca in (1.0, 0.45):
    mm = pyxpp.load(PKG / "examples" / "lecar.ode")
    mm.pars["gca"] = gca
    mm.run(total=400); mm.run_last(total=10)
    e = mm.auto.run("iapp", start="steady", ds=0.005, dsmax=0.01, parmin=-0.2, parmax=0.5, nmax=2000)
    for h in e.special("HB"):
        l1, om = mm.auto.lyapunov(h)
        orb = mm.auto.run(grab=h.label, choice="periodic", ntst=40, ds=0.002, dsmax=0.005, nmax=8)
        print(f"gca = {gca}: Hopf at iapp = {h.x:.4f}, l1 = {l1:+8.3f};  orbits 2-6 stable: {[p.stable for p in orb[2:7]]}")

# %% [markdown]
# **The Brusselator** x' = a - (b+1) x + x^2 y, y' = b x - x^2 y: Hopf on
# b = 1 + a^2.

# %%
bru = pyxpp.load(text="""
x'=a-(b+1)*x+x^2*y
y'=b*x-x^2*y
par a=1,b=1
init x=1,y=1
@ total=50
done
""")
bru.run(); bru.run_last()
b1 = bru.auto.run("b", start="steady", ds=0.01, dsmax=0.05, parmin=0, parmax=5, nmax=300)
hb = b1.special("HB")[0]
print(f"Hopf at b = {hb.x:.6f} for a = 1 (exact 2)")
c1 = bru.auto.run(grab=hb.label, choice="two param", param2="a", ds=0.01, nmax=300, parmin=0, parmax=5)
c2 = bru.auto.run(grab=hb.label, choice="two param", ds=-0.01)
pts = [(p.par["a"], p.x) for p in (*c1, *c2)]
print("max |b - (1 + a^2)| on the Hopf curve:", f"{max(abs(b - 1 - a * a) for a, b in pts):.1e}")

# %% [markdown]
# The curve is as accurate as AUTO's convergence criteria allow: with
# smaller `epsl`, `epsu` (and a smaller step) it lies closer to the exact
# curve. `reset()` clears the diagram first; note that grabbing points and
# running leaves the parameters at the values of the last point.

# %%
bru.auto.reset()
print("after the runs: a =", round(bru.pars["a"], 4), " b =", round(bru.pars["b"], 4), "(AUTO leaves the last point's values)")
bru.set(a=1, b=1, x=1, y=1)
bru.run(); bru.run_last()
tight = dict(epsl=1e-8, epsu=1e-8, epss=1e-6)
b1 = bru.auto.run("b", start="steady", ds=0.01, dsmax=0.05, parmin=0, parmax=5, nmax=300, **tight)
hb = b1.special("HB")[0]
c1 = bru.auto.run(grab=hb.label, choice="two param", param2="a", ds=0.01, dsmax=0.02, nmax=600)
c2 = bru.auto.run(grab=hb.label, choice="two param", ds=-0.01)
pts = [(p.par["a"], p.x) for p in (*c1, *c2)]
print("with epsl = epsu = 1e-8: max |b - (1 + a^2)| =", f"{max(abs(b - 1 - a * a) for a, b in pts):.1e}")
