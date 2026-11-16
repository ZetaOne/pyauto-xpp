# %% [markdown]
# ## Every XPP button from Python
#
# `m.buttons` mirrors XPP's main window menu by menu, with XPP's keyboard
# shortcuts; `m.auto.buttons` mirrors the AUTO window. `pyxpp.show` prints a
# tree (here the start of it; the whole tree has about 230 buttons).

# %%
import io
import contextlib
import warnings
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)

m = pyxpp.load(PKG / "examples" / "lecar.ode")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pyxpp.show(m.buttons)
tree = buf.getvalue().splitlines()
print("\n".join(tree[:22]))
print(f"... {len(tree)} lines in all")
print(m.buttons.numerics.method, " ", m.buttons.numerics.method.cvode)

# %% [markdown]
# **Answers.** A button's positional arguments answer what XPP asks next, in
# order:
#
# | XPP asks | give |
# |---|---|
# | a number or text prompt | the value |
# | a menu | the key letter or (the start of) the entry's label |
# | a dialog with several fields | a dict `{field: value}` (fields not given keep their value) |
# | a file name | the name (relative to `m.workdir`) |
# | a yes/no question | `True` / `False` |
#
# Anything not answered takes XPP's default, as if Enter were pressed.
# `m.transcript` shows each question and the answer used.

# %%
m.buttons.numerics.total(300)                       # a prompt
m.buttons.numerics.method("Runge")                  # a menu, by label
m.buttons.numerics.method("c", 1e-9, 1e-9)          # by key; then CVODE's tolerance prompts
print(m.transcript)
m.buttons.viewaxes.two_d({"X-axis": "v", "Y-axis": "w", "Xmin": -0.6, "Xmax": 0.5})
print(m.transcript)

# %% [markdown]
# **Finding out what a button asks.** Press it with no answers and read the
# transcript; a dialog given a wrong field name lists its fields; a menu
# given a wrong entry lists its entries.

# %%
m.buttons.numerics.sing_pt_ctrl()
print(m.transcript)
for call in (lambda: m.buttons.numerics.poincare_map.section({"?": 0}),
             lambda: m.buttons.numerics.method("nonsense")):
    try:
        call()
    except pyxpp.XPPError as e:
        print("XPPError:", str(e).splitlines()[0])
# A button that fails half way keeps what XPP had already done: here the
# Poincare dialog had already switched the map on, so switch it off again.
m.buttons.numerics.poincare_map.none()

# %% [markdown]
# **Answers that were not asked for** give an `AnswersNotUsed` warning (the
# button asked less than you thought).

# %%
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m.buttons.erase("not asked")
print([str(x.category.__name__) for x in w])

# %% [markdown]
# **Raw keys.** `m.press(key, *answers)` presses one key in the main window,
# as typed in XPP; `m.press("esc")` goes back to the main menu.
# `m.command(id)` runs an internal command number (`M_*` in `menudrive.h`).

# %%
m.press("u")                       # open nUmerics
m.press("esc")
m.press("i", "g")                  # Initialconds > Go
print(m.transcript or "(no questions)", " rows:", len(m.data))

# %% [markdown]
# **The File menu.** Save and read `.set` files (parameters, initial data,
# numerics, graphics), print the source, write a readable summary, clone the
# file, evaluate expressions with the calculator, change right-hand sides.
# Every button that writes a file asks "File Exists! Overwrite?" when the
# file is there already; without the answer `True` it keeps the old file.

# %%
m.set(iapp=0.07, total=120)
m.buttons.file.write_set("run1.set")                # a new file
m.buttons.file.write_set("run1.set", True)          # it exists now: XPP asks "Overwrite?" -> True
m.set(iapp=0.0, total=30)
m.buttons.file.read_set("run1.set")
print("after read_set: iapp =", m.pars["iapp"], " total =", m.numerics["total"])

m.buttons.file.save_info("info.txt")
print((m.workdir / "info.txt").read_text().splitlines()[:8])

m.buttons.file.clone("copy_of_lecar.ode")
print("clone written:", (m.workdir / "copy_of_lecar.ode").exists())

m.buttons.file.calculator("iapp*100", "exp(1)", pyxpp.ESC)   # ESC ends the calculator
print(m.transcript)

# %%
# Edit > RHS's: a dialog whose fields are dV/dT, dW/dT and the aux names;
# the model changes in place
m.buttons.file.edit.rhs({"dW/dT": "0"})
print(m.transcript.strip())
tr = m.run(total=20, v=-0.3, w=0.1)
print("w stays at 0.1 after editing w' = 0:", tr["w"].min(), tr["w"].max())

# %% [markdown]
# **Transcript of anything.** `m.log` holds everything XPP and AUTO printed
# since loading; `m.transcript` the questions of the last call only.

# %%
print(len(m.log.splitlines()), "lines in the log; last:", m.log.strip().splitlines()[-1])
