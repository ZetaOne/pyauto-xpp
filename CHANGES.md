# pyauto-xpp: changes from pyxpp

This folder starts from pyxpp by Youngmin Park
(https://github.com/youngmp/pyxpp, GPL-2.0-or-later; copy in
`../resources/pyxpp`). The Python import name is still `pyxpp` for now.

## 2026-11-17

- **AUTO from a given orbit.** `m.auto.run(param, start="periodic" | "bdry value"
  | "homoclinic" | "heteroclinic", initial=(t, U))` (or `initial=` a
  Trajectory) starts a run from scratch from the given solution instead of
  the data browser. XPP reads the starting orbit from its data storage at
  equally spaced rows and takes the period from the last time, so the data
  are resampled to equal steps if needed and shifted to start at t = 0.
  `m.load_data(t, U)` puts data into the browser on its own (new export
  `pyxpp_load_storage`; XPP computes the auxiliary columns).
- **Stored solutions.** `Point.solution` -> `(t, U)`: the solution AUTO stored
  for a labelled point (the `.s` file), on AUTO's mesh, t normalised to
  [0, 1]; `m.auto.solution(point, real_time=True)` multiplies by the period.
  Periodic orbits of delay equations come from XPP's in-memory store (new
  export `pyxpp_auto_solution`).
- **Double-precision final state.** `m.last_state()` -> `{"t": ..., var: ...}`
  from XPP's own integration state (`MyData`, `MyTime`), not the
  single-precision data browser (new export `pyxpp_last_state`).
- **Fixed: AUTO crashed the process for long meshes.** AUTO wrote each
  solution header in `.s`/fort.8 as `%5ld` fields with no separator; once the
  rows-per-solution count reached 10000 (NTST*NCOL*rows per mesh point, e.g.
  NTST >= 1249 with NCOL 4 for 2 variables) it ran into the previous field,
  the label was then never found on restart ("Restart label not found") and
  XPP read a NULL `FILE*` while appending the missing fort.8. The fields are
  now written `" %4ld"` (identical output below 10000; a `build_lib.py`
  patch, which now takes an optional expected match count). Also patched:
  `appendf`/`copyf` return on a missing file, and File > Import orbit refuses
  an orbit longer than MAXSTOR instead of overflowing the storage. Before a
  restart pyxpp checks the grabbed label is stored and raises `XPPError`
  otherwise; a "Restart label not found" from AUTO also raises `XPPError`.
- Tests for each (39 in all).

## 2026-11-16

- **Windows.** `build_lib.py` builds `pyxpp/_libxpp.dll` with MinGW-w64 gcc
  (natively, or `--target windows` to cross-compile from Linux); all tests
  pass under Wine with python.org CPython 3.11. Windows-only code lives in
  `csrc/win/` (missing POSIX headers; `wincompat.h`, force-included) and an
  `#ifdef _WIN32` block in `headless.c`: `HOME` falls back to `USERPROFILE`;
  XPP's terminal output is captured by redirecting its `stdout`/`stderr`
  streams inside the DLL (the DLL's C runtime is not Python's, so `dup2` from
  Python cannot do it); files XPP writes, and AUTO's scratch files, are opened
  in binary mode (AUTO seeks backwards in `fort.3`, which text mode breaks).
  `_lib.py` keeps each model's DLL copy in `%TEMP%\pyxpp-dll` and unloads and
  deletes it when the model is closed or freed.
- **Fixed: AUTO left `fort.8` open** and then renamed it (a file-descriptor
  leak on Linux; on Windows the rename fails and the next run from a labelled
  point crashed). It is now closed first (a `build_lib.py` patch).
- **XPPAUT 8.2** (Bard's source of 2026-11-15) replaces 8.1 in `csrc/xppaut`.
  New in 8.2 and working through pyxpp unchanged: AUTO continuation of
  equilibria *and periodic orbits of delay equations* (`dde_po.c`), with
  period-doubling and torus detection, and the DDE spectrum (`del_spec.c`).
  `stdfix.h` includes replaced by `<stdlib.h>` as before; `sbml2xpp.c` (a
  separate program) left out; five Xlib calls of the new surface view stubbed
  in `nox11.c`. The library is now replaced in one step when rebuilt.
- **Fixed: `meth=` in ModelBuilder.** XPP reads only the first letter of the
  method, so `dopri5` (and XPP Studio's `dormprince`) silently meant
  *discrete*. ModelBuilder now writes XPP's own name (`5dp`, `rungekutta`, ...)
  and rejects unknown names.
- **Fixed: a second `m.auto.run(start=...)`** after a diagram existed returned
  too few points or none (the old diagram is destroyed and the new points are
  numbered from 0).
- **Fixed: delay equations and dt.** Changing dt (or the delay) after loading
  did not resize XPP's store of past values, which silently broke the delay;
  now rebuilt as XPP's nUmerics dialog does. Volterra kernels likewise.
- **Fixed: Volterra kernels and parameters.** Changing a parameter left the
  kernel table at the old value (XPP re-evaluates only with AutoEval on, off
  by default); pyxpp now always re-evaluates kernels and function tables.
- **Fixed: `equilibrium()`** reported "not found" when Newton converged and the
  eigenvalues were computed but XPP's later eigenvector step failed.
- **Fixed: numpy numbers in ModelBuilder** were written as `np.float64(1.4)`,
  which XPP reads as 0.
- **Fixed (XPP bug, patched in build_lib.py): `@ newt_tol` and `@ jac_eps` in
  an .ode file** were overwritten by the 1e-3 defaults.
- `m.numerics`: `newt_tol` now shows Newton's tolerance (XPP's EVEC_ERR) and
  the new key `jac_eps` the Jacobian step (NEWT_ERR); before, `newt_tol`
  showed the Jacobian step and `evec_tol` the tolerance.
- Seven regression tests added (30 in all).
- **Delay-equation history:** `m.history(x=1.2)` or `m.history(x="formula in t")`
  sets the past (and x(0) to its value at t = 0). A variable without a history
  of its own (none in the file, none given) now has a constant past equal to
  its initial value, so `init x=1.2` / `run(x=1.2)` mean x has always been
  1.2; XPP itself uses a past of 0 there, which silently gave wrong runs.
  `m.has_delays`, `m.calc(expr, t=0)` added.
- **`m.equilibrium()` on a delay equation** returns the rightmost roots of
  the characteristic equation (XPP 8.2's delay spectrum; `nroots=`, default
  12), not the n eigenvalues of a wrong Jacobian.
- **AUTO points: `p.multipliers`** is what AUTO stores (exp(lambda) at an
  equilibrium of a flow, the eigenvalues at a fixed point of a map, Floquet
  multipliers on a periodic orbit). **`p.eigenvalues`** now gives the
  eigenvalues lambda of an equilibrium (computed by XPP at the point); on a
  periodic orbit it raises AttributeError pointing to `.multipliers`.
  (Before, `p.eigenvalues` silently held the multipliers.)
- **`m.auto.run`** raises an XPPError naming the menu when XPP opened a menu
  that got no answer (e.g. extending a periodic branch without
  `choice="extend"`), instead of returning an empty diagram.
- **Two-parameter runs** use epsl = epsu = `Auto.two_par_tol` (1e-7) unless
  given, and restore AUTO's settings afterwards: with AUTO's 1e-4 the
  FitzHugh-Nagumo Hopf curve was off by up to 1.7e-4, now 7e-6 (median 5e-10).
- Five more tests (35 in all).

## 2026-11-14

- AUTO's parameter list (at most 8): asking for a second parameter that is
  not in it, after a diagram exists, used to swap it into the last slot, so
  a grabbed point started the new parameter at another one's value (the
  continuation failed at once). Now a clear XPPError says to include it with
  `m.auto.parameters(...)` before the first run.
- **`Auto.lyapunov(point)` fixed**: it looked the point up by its position
  in AUTO's current diagram, so after `reset()` and another run it used a
  different point (FitzHugh-Nagumo gave l1 = 19 instead of 0.97). Each Point
  now keeps AUTO's full parameter vector and state (`raw_par`, `raw_ubar`)
  and l1 is computed from those (new export `pyxpp_hopf_l1`). Test added.
- XPP's messages on **stderr** (e.g. the out-of-bounds table) now go to
  `m.log` too, instead of leaking into notebooks and terminals.
- Known XPP/AUTO hang: on Wilson-Cowan's sharp folds of equilibria AUTO never
  returns (ignores nmax) when dsmax >= 0.05; dsmax = 0.01 passes. Keep AUTO
  steps small on models with sharp folds.
- **No more leftover temporary folders.** A model's private copy of the
  library (4.5 MB) is deleted as soon as it is loaded (POSIX keeps it
  usable), and a temporary working folder (no `workdir=`) is removed when
  the model is freed or Python exits. Before this, every model left a
  `/tmp/pyxpp_*` folder behind: 2600 of them, 12 GB, had piled up.
- **XPP bug patched (build_lib.py):** nUmerics > stocHast > 2D-hist wrote past
  its buffer when nbins1*nbins2 >= MAXSTOR (the default 100 x 100 with
  maxstor 5000): `twod_hist()` caps the allocation, `two_d_hist()` does not.
  Now XPP's own message "Too many bins" instead of a crash. Present in
  XPP 8.1 itself (worth reporting upstream).
- **Vector fields computed by XPP**: `m.rhs(point)` (XPP's right-hand side
  at a point, new C export `pyxpp_rhs`), `m.vector_field(x, y, window, grid)`
  (that on a mesh, no integration), `m.direction_field(..., scaled=)` (XPP's
  own Dir.field routine with output to a file instead of the screen, new
  export `pyxpp_dirfield`: XPP's arrows, Direct or Scaled). Test added.
- **`Run`** (`pyxpp/run.py`): run conditions as data (numerics, method and
  its settings, parameters, initial conditions, other @ options, start from
  the initial conditions or the last run). `m.run(Run(...))` and
  `m.run_many([...])` apply it for that run only; `m.run(..., keep=False)`
  does the same for keyword settings; `m.using(...)` for a block of runs;
  `m.snapshot()` / `m.restore()`; results carry `.run`; `Run.save/load` (JSON),
  `Run.from_model(m)`. Plain `m.run(**settings)` still keeps its settings.
  Test added (21 pass); usage guide section "Run conditions as data".
- `ModelBuilder(name, outdir=...)`: one folder for the .ode file and XPP's
  working folder (created if missing). `write()` creates missing folders.

## 2026-11-13 (later)

- **XPPAUT 8.1 source of 13 November** (`../resources/xppsrc11_13.tgz`):
  21 files (window-manager close button on every window, bevelled buttons,
  an X error handler, "OK"/"Close" labels; the integrator checks for Esc
  and draws its progress bar about 50 times a second instead of at every
  output point). `csrc/nox11.c` gained `XQueryColor`, `XDrawSegments`,
  `XSetErrorHandler`, `XGetErrorText`. Headless results are identical and
  the speed is the same (the check is a stub here).
- Buttons for 8.1 features that were missing: Initialconds > Periodic;
  Averaging > Isostable, Floquet rho, B correction, H2.
- `Auto.lyapunov(point)`: first Lyapunov coefficient and frequency at a
  Hopf point (8.1's `hopf_l1.c`, new C export `pyxpp_diag_l1`).
  `Auto.gh_points(curve)`: GH (Bautin) points on a two-parameter Hopf curve
  (XPP finds them only while drawing the AUTO window).
- Usage guide: 8.1 periodic start and phase reduction, Lyapunov/GH with a
  check, a section on the XPP versions.
- pyxpp's tests: 20 passed.

## 2026-11-13

- **XPPAUT 8.1 update** (`../resources/update_xppsrc8.1.tgz`, Bard
  Ermentrout, 13 November 2026). Nine files replaced, `stdfix.h` again
  replaced by `<stdlib.h>`:
  - `browse.c`, `browse.h`, `histogram.c`: new `get_browser_rows()`. The
    Fourier transform and histograms use the length of the data shown in
    the browser, not of the stored solution, so they work on the adjoint,
    the H function and other data sets.
  - `arrayplot.c`: the array plot's time range starts at `TRANS` and steps
    by `nout*dt`.
  - `scrngif.c`, `scrngif.h`, `kinescope.c`, `aniparse.c`, `arrayplot.c`:
    every GIF is GIF89a and carries the file name, parameters, date,
    numerics and a note on its contents (a GIF comment, and plain text after
    the end marker).
  - `Makefile`: version 8.1, `extras` and `srconly` targets.
- `build_lib.py`: version 8.1.1 (`MYSTR2=1`), as in Bard's Makefile.
- `Auto.reset()` answers XPP's "destroy?" question only when there is a
  diagram (XPP does not ask otherwise; the unused answer gave a warning).
- AUTO example: `../rungs/AUTO/hh/hh_auto.py` (Hodgkin-Huxley, 49 starts).
- Found while writing the usage guide (`docs/usage/`):
  - `workdir=` given as a relative path failed (XPP changes into it before
    opening its log); it is now made absolute.
  - `m.data` read the stored solution only. It now reads XPP's data browser
    (8.1's `get_browser_rows()`), so a histogram, a Fourier transform, the
    adjoint, H and the Monte-Carlo mean/variance come back as data.
  - Menus whose entries are written "k: name" (File > Get par set, tables)
    accept the name; a wrong menu answer lists the entries.
- `docs/usage/USAGE.mdx`: usage guide, every example run by
  `docs/usage/build_usage.py`.
- pyxpp's tests: 20 passed.

## 2026-11-11

- **XPPAUT 8.1.** `csrc/xppaut/` is Bard Ermentrout's 8.1 source
  (`../resources/new_source_xpp8p1src.tgz`) in place of 8.0, with these
  changes:
  - `#include "stdfix.h"` replaced by `#include <stdlib.h>` in every file,
    and `stdfix.h` removed (it declares libc functions without prototypes;
    the same choice as in XPPAUT-Web).
  - `sbml2xpp.c` removed (a separate converter program that needs libsbml).
  - The macOS `._*` files from the archive removed.
  - pyxpp's source patches (`build_lib.py`) all still apply unchanged.
- `build_lib.py`: version string 8.1; compiler messages decoded with
  `errors="replace"` (8.1 has non-UTF-8 bytes in messages).
- pyxpp's tests: 20 passed with 8.1.
- **New: `pyxpp/builder.py`**, `ModelBuilder`: writes an .ode file
  statement by statement from Python (par, init, number, !, functions, fixed
  quantities, odes, maps, volterra, aux, DAE `0=` and `solv`, markov,
  wiener, tables, special, global events, boundary conditions, export,
  only, `%[..]` blocks, @ options, `set`, tutorial `"` lines, notes after
  `done`, and `raw()` for anything else). Checked against the 214 .ode
  files of the XPPAUT-Web site: see `../scripts/builder-check/REPORT.md`.

## Known problems

- `help/odes/front.ode` makes XPP segfault in about half of the runs (the
  original file as well as the rebuilt one), inside the library.
