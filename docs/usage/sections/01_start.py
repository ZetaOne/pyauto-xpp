# %% [markdown]
# ## Getting started
#
# **Build and install.** The package compiles XPP 8.2 (with AUTO) into a
# shared library the first time it is built. Only a C compiler is needed;
# X11 is not.
#
# ```bash
# cd pyauto-xpp
# python build_lib.py          # builds pyxpp/_libxpp.so
# pip install -e .             # or: put pyauto-xpp on PYTHONPATH
# python -m pytest tests -q    # 20 tests
# ```
#
# **Load a model.** `pyxpp.load` reads an `.ode` file exactly as XPP does,
# including its `@` options, `set` blocks and AUTO settings. The Morris-Lecar
# file shipped with the package is used in many examples below.

# %%
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)

LECAR = PKG / "examples" / "lecar.ode"
m = pyxpp.load(LECAR)
print(m)
print("variables:", m.vars)
print("auxiliary:", m.aux)
print("columns of the data:", m.columns)
print("parameters:", m.par_names)

# %% [markdown]
# Names are kept as XPP stores them (upper case), but every lookup in the
# package ignores case: `m.pars["iapp"]`, `tr["v"]` and `tr["V"]` all work.
#
# **A model from text.** `text=` takes the contents of an `.ode` file; it is
# written into the model's working folder and loaded from there.

# %%
vdp = pyxpp.load(text="""
# van der Pol oscillator
x'=y
y'=-x+mu*(1-x^2)*y
par mu=1
init x=2,y=0
@ total=50,dt=0.01
done
""")
print(vdp, vdp.path.name)

# %% [markdown]
# **Working folder.** Each model has a folder where XPP reads and writes its
# files (`.set` files, AUTO's files, SVGs, ...). By default it is a new
# temporary folder; `workdir=` chooses it. Every call that writes files runs
# with this folder as the current directory, so relative file names land
# there.

# %%
from pathlib import Path

w = pyxpp.load(LECAR, workdir="lecar_work")
print(w.workdir.resolve().relative_to(Path.cwd()))
print(sorted(p.name for p in w.workdir.iterdir()))

# %% [markdown]
# **Several models at once.** Each model runs in its own private copy of the
# XPP library (XPP keeps all its state in C global variables), so models in
# one Python process do not interfere.

# %%
m.pars["iapp"] = 0.05
print("lecar iapp =", m.pars["iapp"], "  vdp has iapp?", "iapp" in vdp.pars)
print("vdp mu =", vdp.pars["mu"], "  lecar has mu?", "mu" in m.pars)

# %% [markdown]
# **XPP's messages.** By default (`quiet=True`) everything XPP and AUTO print
# is collected in `m.log` instead of the terminal. `quiet=False` lets it
# through.

# %%
vdp.run()
print(vdp.log.strip().splitlines()[-3:])

# %% [markdown]
# **Command-line options.** `args=` passes XPP's command-line flags, e.g.
# `-setfile` to read a `.set` file at start (parameters, initial data and
# numerics saved by File > Write set).

# %%
w.set(iapp=0.08, total=150)
w.buttons.file.write_set("start.set")   # save this state (see the buttons section)
w2 = pyxpp.load(LECAR, workdir="lecar_work", args=["-setfile", "start.set"])
print(w2.pars["iapp"], w2.numerics["total"])

# %% [markdown]
# **Closing.** `m.close()` unloads the library copy and frees its memory. A
# model that is no longer referenced is also freed.

# %%
for model in (w, w2):
    model.close()
print("closed")
