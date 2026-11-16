# pyauto-xpp

<img src="site/icon-192.png" width="128" align="right" alt="The pyauto-xpp icon: an orange limit cycle around a blue cubic nullcline, with an unstable equilibrium in the middle, on a dark blue square">

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ZetaOne/pyauto-xpp/HEAD?urlpath=lab/tree/tutorials)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/ZetaOne/pyauto-xpp?quickstart=1)
[![Release](https://img.shields.io/github/v/release/ZetaOne/pyauto-xpp?label=release)](https://github.com/ZetaOne/pyauto-xpp/releases/latest)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.15-blue)](https://github.com/ZetaOne/pyauto-xpp/releases/latest)
[![Platforms](https://img.shields.io/badge/wheels-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/ZetaOne/pyauto-xpp/releases/latest)
[![Licence](https://img.shields.io/badge/licence-GPL--3.0--or--later-green)](LICENSE)

XPPAUT 8.2, the dynamical-systems tool by Bard Ermentrout, and its AUTO as a
Python module. Same numerics, no windows: every XPP button is a Python call,
AUTO runs from Python, and models can be written in Python.

**→ https://zetaone.github.io/pyauto-xpp/**

```python
import pyxpp
m = pyxpp.load("lecar.ode")
tr = m.run(total=100)                       # Initialconds > Go
br = m.auto.run("iapp", start="steady")     # AUTO, from the steady state
```

<p>
  <img src="site/gallery/16_2D-FLOWS_morris-lecar-bifurcation_ml-bifurcation.png" width="49%" alt="Bifurcation diagram of the Morris-Lecar model computed with AUTO from Python: equilibria in the applied current, two Hopf points and the branch of periodic orbits between them">
  <img src="site/gallery/37_DELAYS_mackey-glass-period-doubling_orbit_diagram.png" width="49%" alt="Mackey-Glass delay equation: AUTO's periodic orbits and period doublings drawn over the maxima of simulations">
</p>

*Left: Morris–Lecar in AUTO, equilibria in the applied current i (red stable,
black unstable), the two Hopf points and the periodic orbits between them
(green stable, blue unstable). Right: the Mackey–Glass delay equation, AUTO's
period doublings over the maxima of simulations. Both made by the tutorial
scripts; more in the [gallery](https://zetaone.github.io/pyauto-xpp/#gallery).*

## Features

- **The real XPPAUT 8.2**: Bard Ermentrout's C sources (in `csrc/xppaut`),
  built as a library without X11. Results are XPP's own.
- **Every button is a call**: Initialconds, Sing pts, nullclines, Numerics
  and the rest, as `m.buttons...`, with trajectories returned as NumPy arrays
  and AUTO's diagrams also as pandas tables.
- **AUTO from Python**: steady states, periodic orbits, one- and two-parameter
  diagrams, special points (`HB`, `LP`, `PD`, ...), **also for delay
  equations** (new in XPPAUT 8.2).
- **Models in Python**: `ModelBuilder` writes the `.ode` file for you, or load
  any existing `.ode` file.
- **70 tutorials**: 39 scripts that check their numbers against exact or
  published values, and 31 interactive notebooks with sliders, a direction
  field menu, clickable phase planes and an AUTO panel.
- **Prebuilt** for Python 3.9 to 3.15 on Linux (x86_64, ARM; glibc 2.28+),
  macOS (Apple Silicon 11+; Intel 10.9+ with Python 3.9-3.11, 10.13+ with
  3.12-3.13, 10.15+ with 3.14-3.15) and Windows (x86_64).

## Quick start

| | how | what you need |
|---|---|---|
| **In your browser** | [Open the tutorials in Binder](https://mybinder.org/v2/gh/ZetaOne/pyauto-xpp/HEAD?urlpath=lab/tree/tutorials) | nothing (starts in a minute or two; edits are not kept) |
| **In your browser, with GitHub** | [Open in GitHub Codespaces](https://codespaces.new/ZetaOne/pyauto-xpp?quickstart=1) | a GitHub account; uses its free monthly Codespaces hours. JupyterLab opens by itself; your work is kept until you delete the codespace |
| **pixi** (recommended) | `pixi run lab` in this folder | [pixi](https://pixi.sh), installed once |
| **pip, prebuilt** | `pip install` the wheel from [Releases](https://github.com/ZetaOne/pyauto-xpp/releases/latest) | Python 3.9 to 3.15 on Linux or macOS |
| **pip, from source** | `pip install .` in this folder | a C compiler |

### pixi

Install [pixi](https://pixi.sh) once, then:

```bash
git clone https://github.com/ZetaOne/pyauto-xpp && cd pyauto-xpp
pixi run lab        # JupyterLab with the tutorials opens in your browser
pixi run scripts    # every tutorial script, one line each
pixi run test       # the package's tests
```

The first run downloads Python and the packages and builds XPP's C library;
after an update, only what changed is downloaded again. No git? Download
[the .zip](https://github.com/ZetaOne/pyauto-xpp/archive/refs/heads/main.zip),
unzip it and run the same commands in that folder.

### pip

From the [Releases page](https://github.com/ZetaOne/pyauto-xpp/releases/latest),
download the wheel that matches your system and your Python
(`cp312` = Python 3.12), then:

```bash
python -m pip install pyauto_xpp-0.1.0-cp312-cp312-<your system>.whl
python -m pip install matplotlib     # for the figures in the tutorials
```

| system | the wheel's name ends in |
|---|---|
| Linux x86_64 | `x86_64.whl` |
| Linux ARM | `aarch64.whl` |
| macOS Intel (10.9+, 10.13+ or 10.15+ by Python version) | `x86_64.whl` (with `macosx`) |
| macOS Apple Silicon (11 or newer) | `arm64.whl` |
| Windows x86_64 | `win_amd64.whl` |

Or build it yourself with `python -m pip install .` in this folder (needs a C
compiler: gcc or clang).

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

The full guide is [`docs/usage/USAGE.mdx`](docs/usage/USAGE.mdx).

## Tutorials

| group | scripts | notebooks | examples |
|---|---:|---:|---|
| 2D-FLOWS | 14 | 14 | FitzHugh–Nagumo, Morris–Lecar, Hodgkin–Huxley, toggle switch, forced Duffing |
| EVENTS | 2 | 2 | bouncing ball, integrate-and-fire |
| MAPS | 5 | 5 | logistic, Hénon, standard (Chirikov–Taylor) map |
| 3D-FLOWS | 3 | 3 | Lorenz, Lorenz 84, Rössler |
| DELAYS | 11 | 3 | Mackey–Glass, Hutchinson, delay-coupled Stuart–Landau, two delays |
| NOISE | 2 | 2 | FitzHugh–Nagumo with white noise and with a Markov input |
| SPACE | 2 | 2 | a chain of FitzHugh–Nagumo cells, the Nagumo front |

- [`tutorials/scripts/`](tutorials/scripts): plain Python, one model each.
  Each builds its model in code, analyses it, saves its figures to
  `figures/<script>/` and ends with `N/M checks passed`. In JupyterLab
  (`pixi run lab`, Binder or Codespaces), double-click a script: it opens as a notebook,
  one cell per `# %%`, with the figures inline; edits are saved back into the
  `.py` file.
- [`tutorials/notebooks/`](tutorials/notebooks): interactive notebooks.
  They use `xpptut.py` from the same folder.

The [project page](https://zetaone.github.io/pyauto-xpp/#tutorials) lists
them all with a search, and a View, Download and Open in Binder link for each.

## Sister project

[**XPPAUT-Web**](https://zetaone.github.io/XPPAUT-Web/): the same XPPAUT 8.2
with its own windows, in the browser and as an Android app. Use it to click;
use pyauto-xpp to script.

## Report a bug or give feedback

[Open an issue](https://github.com/ZetaOne/pyauto-xpp/issues) on GitHub.
Please mention your system and Python version, and if a model misbehaves,
attach the `.ode` file (or the `ModelBuilder` code) and any files it uses.

## How it works

XPP's C sources are compiled into a shared library without its X11 interface
(`csrc/headless.c`, `csrc/nox11.c`), which Python loads with `ctypes`
(`pyxpp/_lib.py`). The Python side (`pyxpp/`) presses XPP's buttons and
answers its prompts for you, and reads back the trajectories, equilibria and
AUTO's diagrams. Changes are listed in [`CHANGES.md`](CHANGES.md).

## Credits and licence

**GNU General Public License, version 3 or later** (GPL-3.0-or-later). See
[`LICENSE`](LICENSE).

- **XPPAUT** is by G. Bard Ermentrout. The C sources in `csrc/xppaut` are his
  version 8.2, including AUTO for delay equations.
  [Homepage](https://sites.pitt.edu/~phase/bard/bardware/xpp/xpp.html).
- **AUTO** is by E. Doedel and collaborators.
- **pyxpp**, the headless Python bridge this package grew from, is by Youngmin
  Park.
- **The tutorial models** come from the examples of XPP Studio (Peter Ashwin and
  Bard Ermentrout, GPL-3.0) and from the delay-equation examples that come with
  XPPAUT 8.2.
