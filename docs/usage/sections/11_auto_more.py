# %% [markdown]
# ## AUTO III: branch points, periodic starts, period doubling, maps
#
# **Branch points.** In the pitchfork x' = mu x - x^3 the trivial branch has
# a branch point (BP) at mu = 0. XPP's AUTO switches branches by itself, so
# one run gives the trivial branch and both new branches (x = +-sqrt(mu)).
# (A BP can also be grabbed with `choice="switch"`; on this example that
# only followed the trivial branch again, also with automatic switching
# off, `mxbf=0`, so rely on the automatic switching.)

# %%
import numpy as np
import matplotlib.pyplot as plt
import pyxpp

pf = pyxpp.load(text="""
x'=mu*x-x^3
par mu=-1
init x=0
@ total=50
done
""")
pf.run(); pf.run_last()
br = pf.auto.run("mu", start="steady", ds=0.01, dsmax=0.05, parmin=-1, parmax=1, nmax=200)
bp = br.special("BP")[0]
print(br, f"  BP at mu = {bp.x:.1e}")
for b in br.branches():
    mu = np.array([p.x for p in b]); x = np.array([p.u0["x"] for p in b])
    sel = mu > 0.01
    err = np.abs(np.abs(x[sel]) - np.sqrt(mu[sel])).max() if abs(x).max() > 0.1 else np.abs(x).max()
    print(f"  branch {b[0].branch}: {len(b)} points, mu {mu.min():.2f}..{mu.max():.2f}, error from the exact branch {err:.1e}")

# %% [markdown]
# **Starting from a periodic orbit found by simulation** (there is no Hopf
# point to start from). Integrate past the transient, measure the period,
# integrate exactly one period, then `start="periodic"`: AUTO reads the
# orbit and its period from the data browser. Then grab each period-doubling
# point (PD) and choose `"doubling"`. Roessler system, parameter c.

# %%
r = pyxpp.load(text="""
x'=-y-z
y'=x+a*y
z'=b+z*(x-c)
par a=0.2,b=0.2,c=2.5
init x=1,y=1,z=0
@ total=1000,dt=0.005,trans=900,maxstor=100000
done
""")
tr = r.run()
y, t = tr["y"], tr.t
up = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]


def crossing(k):
    s = -y[k] / (y[k + 1] - y[k])
    return t[k] + s * (t[k + 1] - t[k]), {v: float(tr[v][k] + s * (tr[v][k + 1] - tr[v][k])) for v in r.vars}


t1, start = crossing(up[-2])
t2, _ = crossing(up[-1])
r.set(trans=0, total=t2 - t1, **start)
r.run()                                            # one period, starting on the section y = 0
p1 = r.auto.run("c", start="periodic", ntst=50, ds=0.01, dsmax=0.05, nmax=300, parmin=2, parmax=4.5)
pds, branch = [], p1
for k in range(3):
    if not branch.special("PD"):
        break
    pd = branch.special("PD")[0]
    pds.append(pd.x)
    print(f"PD {k + 1}: c = {pd.x:.5f}, period {pd.period:.4f}")
    branch = r.auto.run(grab=pd.label, choice="doubling", nmax=300)
print("ratio of successive PD spacings:", round((pds[1] - pds[0]) / (pds[2] - pds[1]), 3), "(Feigenbaum's limit 4.669)")
fig, ax = plt.subplots(figsize=(7, 4))
r.auto.diagram.plot(y="x", ax=ax)
ax.set(xlim=(2, 4.5), title="Roessler: period doubling in c (max and min of x)")
fig.tight_layout(); fig.savefig(FIG / "roessler_pd.png", dpi=110)

# %% [markdown]
# **Other choices at a point.**
#
# | grabbed point | `choice=` |
# |---|---|
# | HB | `periodic`, `extend`, `new point`, `two param` |
# | BP | `switch`, `extend`, `new point`, `two param` |
# | LP | (none: two-parameter run) |
# | PD | `doubling`, `two param`, `fixed period`, `extend` |
# | TR | `two param`, `fixed period`, `extend` |
# | a point on a periodic branch | `extend`, `fixed period` |
#
# `extend` carries on from the end of a branch; `fixed period` follows the
# orbit with its period held fixed in two parameters. A point is grabbed by
# its label, by a `Point`, or as `"#k"` / `-1` for the k-th / last point.

# %%
ext = r.auto.run(grab=p1[-1], choice="extend", nmax=50, parmax=6)
print("extend the period-1 branch past its end (c = 4.5) with parmax = 6:", len(ext),
      "more points, c up to", round(max(p.x for p in ext), 4))

# %% [markdown]
# **Maps.** For a difference equation (`meth=discrete`) AUTO follows fixed
# points; a 2-cycle is a fixed point of the second iterate. Logistic map
# x -> r x (1 - x): the fixed point loses stability at r = 3, where its
# multiplier passes -1 (a flip). XPP labels that point HB (AUTO's type 3);
# a label with no type is a regular output point (every `npr` steps).

# %%
f1 = pyxpp.load(text="""
x(t+1)=r*x*(1-x)
par r=2.8
init x=0.6
@ meth=discrete,total=200
done
""")
f1.run(); f1.run_last()
m1 = f1.auto.run("r", start="steady", ds=0.01, dsmax=0.02, parmin=2.5, parmax=4, nmax=200)
print("fixed point:", [(p.type, round(p.x, 5)) for p in m1.labeled()])
print("multiplier at the HB point:", round(m1.special("HB")[0].eigenvalues[0].real, 5), " exact 2 - r =", round(2 - m1.special("HB")[0].x, 5))
f2 = pyxpp.load(text="""
f(x)=r*x*(1-x)
x(t+1)=f(f(x))
par r=3.2
init x=0.5
@ meth=discrete,total=400
done
""")
f2.run(); f2.run_last()
m2 = f2.auto.run("r", start="steady", ds=0.01, dsmax=0.02, parmin=2.9, parmax=4, nmax=200)
print("2-cycle:", [(p.type, round(p.x, 5)) for p in m2.labeled()], " (exact PD of the 2-cycle: 1 + sqrt 6 =", round(1 + 6 ** 0.5, 5), ")")
