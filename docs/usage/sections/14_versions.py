# %% [markdown]
# ## The XPP inside: versions and what each source drop changed
#
# The package carries Bard Ermentrout's XPPAUT 8.2 source (`csrc/xppaut/`),
# updated as new drops arrive; `CHANGES.md` lists each one. The version XPP
# reports is in the log.

# %%
import time
from pathlib import Path
import pyxpp

PKG = Path(pyxpp.__file__).resolve().parents[1]   # the pyauto-xpp folder (examples/lecar.ode is in it)

m = pyxpp.load(PKG / "examples" / "lecar.ode")
print(next(l for l in m.log.splitlines() if "XPPAUT" in l).strip())

# %% [markdown]
# | source | what changed | in pyauto-xpp |
# |---|---|---|
# | 8.1 (11 Nov 2026) | Initialconds > Periodic; phase reduction in Averaging (isostable I, Floquet rho, B correction, H2); subcritical Hopf curves and GH points in AUTO | `buttons.initialconds.periodic`, `buttons.numerics.averaging.isostable` / `floquet_rho` / `b_correction` / `h2` (section "Analysis tools"); `auto.lyapunov(point)` and `auto.gh_points(curve)` (section "AUTO II": XPP finds GH points while drawing, so headless they are found in Python the same way) |
# | update (13 Nov 2026) | every GIF carries the model, parameters, numerics and a note; Fourier and histograms work on whatever the data browser shows (`get_browser_rows`) | `m.data` reads the browser, so the histogram, Fourier, adjoint, H come back as data (section "Analysis tools") |
# | 8.2 (15 Nov 2026) | AUTO for delay equations: equilibria and periodic orbits (`dde_po.c`), with period-doubling and torus detection and two-parameter curves; the characteristic roots of delay equations (`del_spec.c`); a rotatable surface view | `m.auto.run` works on delay equations unchanged; `m.equilibrium()` returns the rightmost characteristic roots; `m.history()` |
# | 13 Nov 2026, later | window-manager close button on every window, bevelled buttons, an X error handler, "OK"/"Close" labels; the integrator draws its progress bar and checks for Esc about 50 times a second instead of at every output point | no window, so nothing to see; results identical; same speed (below) |
#
# **The integrator change, timed.** In the X11 GUI each progress/Esc check
# flushes the X connection, so checking 50 times a second instead of at
# every stored point speeds up long runs there. Headless the check is a
# stub, so the speed is the same; the numbers are identical.

# %%
for label, kw in (("rk4, dt = 0.05, every step stored", dict(meth="rk4", dt=0.05, nout=1)),
                  ("rk4, dt = 0.05, every 20th step stored", dict(meth="rk4", dt=0.05, nout=20))):
    best = min((lambda t0: (m.run(iapp=0.1, total=10000, v=-0.3, w=0, **kw), time.perf_counter() - t0)[1])(time.perf_counter())
               for _ in range(3))
    print(f"{label:40s} {best * 1000:6.0f} ms for {len(m.data):6d} rows, v at t = 10000: {m.data['v'][-1]:.7f}")
