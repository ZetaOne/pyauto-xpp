# pyauto-xpp

XPPAUT 8.2 and its AUTO as a Python module (`import pyxpp`), with tutorials.
No X11: every XPP button is a Python call, AUTO runs from Python, and models
can be written in Python with `ModelBuilder`.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ZetaOne/pyauto-xpp/HEAD?urlpath=lab/tree/tutorials)
← try it in your browser, nothing to install (starts in a minute or two; edits are not kept)

## On your own computer

Install [pixi](https://pixi.sh) once, then in this folder:

```
pixi run lab        # JupyterLab opens in your browser, running on your computer
pixi run scripts    # run all tutorial scripts, one line each
pixi run test       # the package's tests
```

The first run downloads Python and the packages and builds XPP's C library;
after that, an update downloads only what changed.

Without pixi: `pip install .` in this folder (needs a C compiler), or a
prebuilt wheel from the Releases page once they are published.

Linux and macOS are supported. Windows support is in progress.

## Tutorials

- `tutorials/notebooks/`: 31 interactive notebooks with sliders, a direction-field
  menu, clickable phase planes and an AUTO panel.
- `tutorials/scripts/`: 39 plain Python scripts, one per model:
  - grouped as 2D-FLOWS, EVENTS, MAPS, 3D-FLOWS, DELAYS, NOISE and SPACE;
  - each builds its model in code, analyses it and checks its numbers against
    exact or published values;
  - in JupyterLab, right-click a script and choose *Open With > Jupytext
    Notebook* to run it cell by cell.

## A first look

```python
import pyxpp
from pyxpp.builder import ModelBuilder

mb = (ModelBuilder("fhn")
      .ode("v", "v-v^3/3-w+i").ode("w", "eps*(v+a-b*w)")
      .parameter(i=0.2, a=0.7, b=0.8, eps=0.08).initial(v=-1.2, w=-0.6))
m = mb.load()
tr = m.run(total=200)                         # Initialconds > Go
m.equilibrium({"v": -1.2, "w": -0.6})         # Newton + eigenvalues
m.run(total=300); m.run_last()
d = m.auto.run("i", start="steady", ds=0.01, parmin=0, parmax=2)   # AUTO
d.special("HB")
```

## Credits and licence

- **XPPAUT** is by G. Bard Ermentrout. The C sources in `csrc/xppaut` are his
  version 8.2, including AUTO for delay equations.
- **AUTO** is by E. Doedel and collaborators.
- **pyxpp**, the headless Python bridge this package grew from, is by Youngmin
  Park.
- **The tutorial models** come from the examples of XPP Studio (Peter Ashwin and
  Bard Ermentrout, GPL-3.0) and from the delay-equation examples that come with
  XPPAUT 8.2.

GPL. See `LICENSE`.
