# %% [markdown]
# ## Writing models with ModelBuilder
#
# `ModelBuilder` writes an `.ode` file one statement at a time from Python.
# Statements come out in the order the methods are called, because order
# matters in an `.ode` file. `render()` shows the text, `write()` saves it,
# and `load()` saves it and loads it into XPP in one step.

# %%
from pyxpp.builder import ModelBuilder

ml = (ModelBuilder("lecar")
      .comment("Morris-Lecar")
      .parameter(iapp=0.1, phi=0.333)
      .parameter(v1=-0.01, v2=0.15, v3=0.1, v4=0.145, gca=1)
      .parameter(vk=-0.7, vl=-0.5, gk=2.0, gl=0.5)
      .function("minf(v)", ".5*(1+tanh((v-v1)/v2))")
      .function("ninf(v)", ".5*(1+tanh((v-v3)/v4))")
      .function("lamn(v)", "phi*cosh((v-v3)/(2*v4))")
      .fixed("ica", "gca*minf(v)*(v-1)")
      .ode("v", "iapp+gl*(vl-v)+gk*w*(vk-v)-ica")
      .ode("w", "lamn(v)*(ninf(v)-w)")
      .initial(v=-0.3, w=0)
      .aux("ica", "ica")
      .option(total=100, dt=0.05, xp="v", yp="w", xlo=-0.6, xhi=0.6, ylo=-0.1, yhi=0.6)
      .parameter_set("hopf", v3=0, v4=0.3, gca=1.1, phi=0.2)
      .tutorial("{iapp=0.1,total=200} Integrate and see the limit cycle.")
      .notes("Free text after done is ignored by XPP."))
print(ml.render())

# %%
m = ml.load("lecar_built.ode")         # write + pyxpp.load
tr = m.run()
print(m, " v at the end:", round(tr["v"][-1], 4))
print("the @ options written:", ml.options())

# %% [markdown]
# **Where the files go.** `ModelBuilder(name, outdir="output")` puts the .ode
# file (`output/<name>.ode`) and XPP's working folder (its .set files, SVGs,
# AUTO files) in one folder, created if missing. Without `outdir` the .ode
# file goes to the current folder and each model gets a fresh temporary
# working folder; `write(path)` / `load(path, workdir=...)` choose them
# yourself (missing folders are created).

# %%
mo = ModelBuilder("decay_out", outdir="output").ode("x", "-x").initial(x=1)
md_ = mo.load()
md_.buttons.file.write_set("run.set")
print(md_.path.parent.name, "/", md_.path.name, "  workdir:", md_.workdir.name,
      "  files:", sorted(p.name for p in md_.workdir.iterdir() if not p.name.startswith(".")))

# %% [markdown]
# **Numbers and derived quantities.** `number` is a constant that is not a
# parameter; `derived` (`!name=...`) is recomputed whenever a parameter
# changes; `initial_value` writes `x(0)=...`. XPP 8.1 takes a number there
# for a single variable (a formula such as `exp(1)` gives 0), and a formula
# in `[j]` for an array.

# %%
import math

b = (ModelBuilder("decay")
     .number(k0=2)
     .parameter(tau=5)
     .derived("rate", "k0/tau")
     .ode("x", "-rate*x")
     .initial_value("x", 2.5)
     .ode("y[1..3]", "-y[j]")
     .initial_value("y[1..3]", "[j]*2")
     .option(total=5))
print(b.render())
d = b.load("decay.ode")
tr = d.run()
print("x(0) =", tr["x"][0], " x(5) =", round(float(tr["x"][-1]), 5), " exact 2.5*exp(-2) =", round(2.5 * math.exp(-2), 5))
print("y(0) =", [float(tr[f"y{i}"][0]) for i in (1, 2, 3)])

# %% [markdown]
# **Maps** (difference equations) use `map()` and the discrete method.

# %%
lm = (ModelBuilder("logistic")
      .parameter(r=3.2)
      .map("x", "r*x*(1-x)")
      .initial(x=0.2)
      .option(meth="discrete", total=200))
tr = lm.load("logistic.ode").run()
print("last four iterates:", [round(float(v), 4) for v in tr["x"][-4:]], " (a 2-cycle)")

# %% [markdown]
# **Arrays.** XPP's `name[lo..hi]` notation expands to one equation per
# index; `[j]` is the index as a number and `u[j]` the variable. Names that are not Python keywords are passed as
# a dict.

# %%
chain = (ModelBuilder("chain")
         .parameter(d=0.5)
         .ode("u[1..5]", "-u[j]+d*[j]")
         .initial({"u[1..5]": 1})
         .option(total=3))
print(chain.render())
c = chain.load("chain.ode")
print(c.vars, " u5(3) =", round(float(c.run()["u5"][-1]), 4), " exact:", round(2.5 + (1 - 2.5) * __import__("math").exp(-3), 4))

# %% [markdown]
# **Repeated blocks.** `begin_block("1..3")` ... `end_block()` repeats the
# statements between them with `[j]` replaced by 1, 2, 3.

# %%
blk = (ModelBuilder("blocks")
       .begin_block("1..3")
       .ode("x[j]", "-[j]*x[j]")
       .initial({"x[j]": 1})
       .end_block()
       .option(total=1))
print(blk.render())
tr = blk.load("blocks.ode").run()
print("x1, x2, x3 at t=1:", [round(float(tr[f"x{i}"][-1]), 5) for i in (1, 2, 3)],
      " exact:", [round(math.exp(-i), 5) for i in (1, 2, 3)])

# %% [markdown]
# **Events.** `global_event(sign, condition, actions)` resets variables when
# a quantity crosses zero (here an integrate-and-fire neuron).

# %%
lif = (ModelBuilder("lif")
       .parameter(I=1.5, vth=1, vreset=0)
       .ode("v", "-v+I")
       .global_event(1, "v-vth", {"v": "vreset"})
       .aux("spike", "v")
       .option(total=10, dt=0.001))
tr = lif.load("lif.ode").run()
resets = tr.t[1:][tr["v"][1:] < tr["v"][:-1] - 0.5]       # v drops at each reset
print("max v =", round(float(tr["v"].max()), 4), " period:", round(float(resets[1] - resets[0]), 3),
      " exact ln(1.5/0.5) =", round(math.log(3), 3))

# %% [markdown]
# **Noise and Markov chains.** `wiener` adds white noise; `markov` defines a
# discrete state with transition rates (row k: rates out of state k).

# %%
sde = (ModelBuilder("ou")
       .parameter(sig=0.3)
       .wiener("w")
       .ode("x", "-x+sig*w")
       .option(total=2000, dt=0.01, meth="euler"))
tr = sde.load("ou.ode").run()
print("OU: variance of x =", round(float(tr["x"][2000:].var()), 4), " theory sig^2/2 =", 0.3 ** 2 / 2)

chan = (ModelBuilder("channel")
        .parameter(a=1, b=2)
        .markov("z", [["0", "a"], ["b", "0"]])
        .ode("q", "z-q")
        .option(total=5000, dt=0.01))
tr = chan.load("channel.ode").run()
print("two-state channel: fraction open =", round(float(tr["z"].mean()), 3), " theory a/(a+b) =", round(1 / 3, 3))

# %% [markdown]
# **Tables, DAEs, integral equations, boundary conditions, special
# operators.** All are written as XPP expects them:
#
# - `table(name, n=, lo=, hi=, expr=)` a tabulated function (or `table(name, file)`);
# - `algebraic` + `solve` for a differential-algebraic system;
# - `volterra` for an integral equation (`int{...}`);
# - `boundary` for a boundary-value problem;
# - `special` for `conv`, `sparse`, `mmult`, ... ;
# - `export`, `only`, `raw` for the rest.

# %%
dae = (ModelBuilder("dae")
       .table("sq", n=101, lo=0, hi=2, expr="t^2")
       .ode("x", "-x+y")
       .solve("y", 1)
       .algebraic("y^3+y-x-sq(1)")
       .aux("yy", "y")                  # a solv variable is not stored; aux keeps it
       .initial(x=0)
       .option(total=2))
tr = dae.load("dae.ode").run()
y, x = tr["yy"][-1], tr["x"][-1]
print("DAE: y^3+y-x-1 at the end =", f"{y**3 + y - x - 1:.1e}")

volt = (ModelBuilder("volterra")
        .volterra("u", "1-int{exp(-t)#u}")
        .option(total=2, meth="volterra", dt=0.01))
print(volt.render())
tr = volt.load("volt.ode").run()
print("Volterra: u(2) =", round(tr["u"][-1], 5), " exact 1/2+exp(-4)/2 =", round(0.5 + 0.5 * __import__("math").exp(-4), 5))

bvp = (ModelBuilder("bvp")
       .ode("x", "y")
       .ode("y", "-x")
       .boundary("x")
       .boundary("x'-1")
       .option(total=1.5707963))
print(bvp.render())

conv = (ModelBuilder("conv")
        .table("w", n=11, lo=-5, hi=5, expr="exp(-abs(t))")
        .ode("u[0..20]", "-u[j]+k([j])")      # network values are k(0), k(1), ...
        .special("k", "conv(even,21,5,w,u0)")
        .initial({"u[0..20]": 0.1})
        .only("u0", "u10")
        .raw("# raw() writes any line as it is")
        .option(total=1))
print(conv.render())
print("loads:", conv.load("conv.ode").vars[:3], "...")

# %% [markdown]
# **Statements written so far.** `lines` holds them; `raw_count` counts the
# `raw()` lines (a model that needs `raw()` shows a gap in the builder).

# %%
print(len(conv.lines), "lines, raw:", conv.raw_count, " ", repr(conv))
