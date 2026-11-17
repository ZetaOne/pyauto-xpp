# %% [markdown]
# ## Errors, limits and tips
#
# **Errors.** Anything XPP refuses raises `pyxpp.XPPError` with XPP's
# message (and, for a button, the transcript). XPP's `exit()` is turned
# into an error, so Python keeps running; a loop that keeps asking with no
# answer stops after 500 questions.

# %%
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)

for what, call in (
        ("a bad .ode file", lambda: pyxpp.load(text="x'=y+\ndone\n")),
        ("File > Quit", lambda: pyxpp.load(PKG / "examples" / "lecar.ode").buttons.file.quit(True)),
        ("a button that keeps asking", lambda: pyxpp.load(PKG / "examples" / "lecar.ode").buttons.file.tutorial()),
        ("an unknown parameter", lambda: pyxpp.load(PKG / "examples" / "lecar.ode").pars.__setitem__("nope", 1)),
        ("AUTO with nothing to start from", lambda: pyxpp.load(PKG / "examples" / "lecar.ode").auto.run("iapp"))):
    try:
        call()
    except (pyxpp.XPPError, KeyError, ValueError) as e:
        print(f"{what:32s} -> {type(e).__name__}: {str(e).splitlines()[0][:90]}")

# %% [markdown]
# **Limits** (headless XPP):
#
# - Buttons that need a live screen do nothing useful: Window > Scroll,
#   AUTO Axes > Scroll, Viewaxes > Toon, Kinescope (blank frames).
# - Mouse buttons (Initialconds > Mouse, zoom boxes, text placement) take
#   pixel answers `"ix,iy"` / `"x1,y1,x2,y2"`.
# - Storage is not a limit: XPP (8.1 and later) enlarges it when a run needs more rows
#   than `maxstor` (a 200 001-row run works with the default 5000).
# - A crash inside XPP's C code (a segfault) ends the Python process. Run
#   doubtful models in a subprocess.
#
# **Tips.**
#
# - Start AUTO on an equilibrium: `run()` then `run_last()` before
#   `start="steady"`; for orbits, integrate exactly one period before
#   `start="periodic"`.
# - Writing a file that exists makes XPP ask "Overwrite?": answer `True`
#   (e.g. `write_set("a.set", True)`), or the old file stays.
# - Grabbing and running leaves parameters at the last point's values;
#   set them again before a new start.
# - To learn what a button asks, call it with no answers and read
#   `m.transcript`; a wrong dialog field or menu entry lists the right ones.
# - XPP stores data in single precision: compare stored values to about
#   1e-7 relative.
# - Every model has its own XPP; to run many in parallel, use processes
#   (e.g. `concurrent.futures.ProcessPoolExecutor`), one model each.

# %%
import concurrent.futures as cf


def end_v(iapp):
    m = pyxpp.load(PKG / "examples" / "lecar.ode")
    return float(m.run(iapp=iapp, total=200)["v"][-1])


with cf.ThreadPoolExecutor(1) as ex:          # (processes in real use; one thread here)
    print([round(x, 4) for x in ex.map(end_v, (0.0, 0.05, 0.1))])
