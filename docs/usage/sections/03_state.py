# %% [markdown]
# ## Parameters, initial conditions and options
#
# `m.pars` and `m.ics` behave like dictionaries (names in any case).

# %%
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)

m = pyxpp.load(PKG / "examples" / "lecar.ode")
print(m.pars)
print(m.ics)
m.pars["iapp"] = 0.08
m.ics["v"] = -0.3
m.pars.update(gca=1.1, phi=0.25)
print(m.pars["IAPP"], m.ics["V"], "gca" in m.pars, list(m.ics.keys()))
print(m.pars.as_dict()["gca"], dict(m.ics.items()))

# %% [markdown]
# **`set()` does all three at once**: each name is tried as a parameter,
# then as a variable (initial condition), then as an `@` option, exactly as
# it would be written on an `@` line.

# %%
m.set(iapp=0.1, v=-0.25, w=0.05, total=250, dt=0.02, meth="rk4", maxstor=50000)
print(m.numerics)

# %% [markdown]
# `set_option(name, value)` is the `@`-option part on its own; `get(name)`
# reads any parameter, initial condition or derived quantity.

# %%
m.set_option("bounds", 1000)
m.set_option("nout", 2)
print("bound:", m.numerics["bound"], " nout:", m.numerics["nout"], " get('phi'):", m.get("phi"), " get('v'):", m.get("v"))

# %% [markdown]
# **Integration methods.** `meth=` takes XPP's names and some aliases:

# %%
from pyxpp.model import METHODS, method_key

print(METHODS)
for name in ("rk4", "cvode", "dopri5", "stiff", "euler", "discrete", "gear", "rosen23", "symplectic"):
    m.set(meth=name)
    print(f"{name:10s} -> key {method_key(name)!r} -> {m.numerics['meth']}")

# %% [markdown]
# **Parameter sets** written in the file as `set name {...}` are chosen with
# File > Get par set (here `hopf` from `lecar.ode`).

# %%
m.buttons.file.get_par_set("hopf")
print({k: m.pars[k] for k in ("v3", "v4", "gca", "phi")})
