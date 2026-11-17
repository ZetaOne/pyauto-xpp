# Worked examples: pyauto-xpp in notebooks

Thirty-nine models, from a first phase plane to AUTO on delay equations, in
seven Jupyter notebooks, one per level. They use XPPAUT 8.2 and its AUTO
through pyauto-xpp (`pyxpp`). No widgets: each model is built in code with
`ModelBuilder`, simulated and analysed, drawn with matplotlib, and its numbers
are checked against exact or published values.

Each model has its own section:

- an introduction with the equations, the parameter values and what to look for,
- the steps, each a short explanation followed by the code,
- the checks, with the formulas they compare against, printed as `ok`/`FAIL`.

The notebooks are saved with their outputs, so they can be read without
running them (on GitHub too). To run them yourself:

```
conda activate pyxpp     # your environment
jupyter lab 01_first-steps.ipynb      # one notebook, cell by cell
python run_all.py                     # all notebooks, one line each
python run_all.py 03 06b              # only some
python run_all.py --save              # also store the new outputs in the notebooks
```

`run_all.py` runs the notebooks on the default Python 3 kernel. Every
notebook writes its model files to `odes/<section>.ode` and its figures to
`figures/<section>/`, and its last cell prints `N/M checks passed`.

The models come from two sources:

- the examples of XPP Studio (Peter Ashwin and Bard Ermentrout, GPL-3.0);
- the delay-equation examples that come with XPPAUT 8.2.

Work through them in order: each level builds on the one before.

## Level 1 — First steps: [`01_first-steps.ipynb`](01_first-steps.ipynb)

Simulate, read a phase plane, find equilibria, handle events, iterate simple maps, meet chaos.

| | model | what it shows |
|---|---|---|
| 01 | FitzHugh–Nagumo | phase plane, excitability, Hopf points (exact), periodic orbits |
| 02 | genetic toggle switch | bistability, basins, folds and the cusp in (a1, a2) |
| 03 | bouncing ball | event times, restitution, Zeno time |
| 04 | integrate-and-fire with adaptation | exact intervals, f–I curves, rheobase |
| 05 | logistic map | orbit diagram, Liapunov exponent, period doubling (exact points) |
| 06 | Hénon map | strange attractor, both Liapunov exponents |
| 07 | Lorenz | pitchfork and Hopf (exact), Liapunov exponent, Lorenz map |

## Level 2 — Bifurcations with AUTO: [`02_bifurcations-with-auto.ipynb`](02_bifurcations-with-auto.ipynb)

Follow equilibria and periodic orbits in one parameter: Hopf, folds, period doubling.

| | model | what it shows |
|---|---|---|
| 08 | Wilson–Cowan | S-shaped branch with folds, Hopf in the inhibitory time constant |
| 09 | Morris–Lecar (Hopf regime) | f–I curve, class II onset, hysteresis |
| 10 | Hodgkin–Huxley | period, Hopf points, fold of orbits, bistability |
| 11 | van der Pol, mu = 1000 | stiff solvers, relaxation-oscillation period |
| 12 | driven pendulum | rest vs rotation, hysteresis, Melnikov onset |
| 13 | delayed logistic map | Neimark–Sacker at a = 2, invariant curve |
| 14 | Rössler | equilibria, period doubling in c |
| 15 | Lorenz 84 | folds and Hopf of the westerly flow, Hadley state |

## Level 3 — Two parameters and global bifurcations: [`03_two-parameters-and-global-bifurcations.ipynb`](03_two-parameters-and-global-bifurcations.ipynb)

Curves of Hopf and fold points; homoclinic and heteroclinic orbits.

| | model | what it shows |
|---|---|---|
| 16 | Morris–Lecar | one- and two-parameter diagrams, GH points |
| 17 | Morris–Lecar with a saddle | period blow-up at a homoclinic orbit |
| 18 | interacting saddle-nodes | equilibria from a quartic, folds, Hopf at mu = 0 |
| 19 | Nagumo front | front speed by shooting vs (1 - 2a)/sqrt(2) |

## Level 4 — Forcing, tori and phase: [`04_forcing-tori-and-phase.ipynb`](04_forcing-tori-and-phase.ipynb)

Forced oscillators, maps on a torus, locking, phase reduction.

| | model | what it shows |
|---|---|---|
| 20 | forced Duffing | stroboscopic map, period doubling to chaos |
| 21 | forced phase oscillator | devil's staircase, Arnold tongues |
| 22 | phase oscillators with a table | locking band from the tabulated coupling |
| 23 | standard map | invariant curves vs chaos, area preservation |
| 24 | piecewise affine map | periods and islands at the golden mean |
| 25 | Ginzburg–Landau oscillator | adjoint, H and H2 against exact formulas |

## Level 5 — Noise and space: [`05_noise-and-space.ipynb`](05_noise-and-space.ipynb)

Stochastic and Markov inputs; chains of coupled cells.

| | model | what it shows |
|---|---|---|
| 26 | FitzHugh–Nagumo with white noise | coherence resonance (seeded, reproducible) |
| 27 | FitzHugh–Nagumo with a Markov input | dwell times, fraction on, firing only when on |
| 28 | chain of 20 FitzHugh–Nagumo cells | travelling spikes, speed ~ sqrt(d) |

## Level 6 — Delay equations

From simulations of delay equations to AUTO on them (XPP 8.2). Two notebooks.

### [`06a_delay-equations-simulation.ipynb`](06a_delay-equations-simulation.ipynb)

| | model | what it shows |
|---|---|---|
| 29 | Hutchinson equation | Hopf at r tau = pi/2 from simulations |
| 30 | Mackey–Glass | maxima against tau, delay embedding |
| 31 | gamma-kernel Volterra equation | Volterra vs linear-chain ODE, onset at r tau = 2 |

### [`06b_delay-equations-with-auto.ipynb`](06b_delay-equations-with-auto.ipynb)

| | model | what it shows |
|---|---|---|
| 32 | equation with two delays | Hopf points with two delays |
| 33 | Hutchinson equation | AUTO on a delay equation: Hopf and periodic orbits |
| 34 | cusp with a delay | folds and delay-induced Hopf points |
| 35 | two delay-coupled neurons | Hopf points and the (a, tau) Hopf curve |
| 36 | Wilson–Cowan with delay | (P, tau) Hopf curve |
| 37 | Mackey–Glass | period-doubling cascade with AUTO |
| 38 | delayed Stuart–Landau | rotating waves and their folds |
| 39 | two delay-coupled Stuart–Landau | torus bifurcations of the in-phase wave |

## Things worth knowing (learned while writing these)

- **Single precision:** XPP stores trajectories in single precision, so stored
  values are good to about 1e-7.
- **Equilibrium accuracy:** for accurate equilibria and eigenvalues set
  `m.set(newt_tol=1e-10, jac_eps=1e-8)`. The default Jacobian step (1e-3)
  puts errors of about 1e-3 into eigenvalues.
- **Delayed values:** these act as if the delay were about dt/2 longer, so use
  a small dt near a delay-induced onset.
- **Delay equations:** a variable's past follows its initial value unless you
  give one (`m.history(x="0.5+0.2*sin(3*t)")`, or `x(0)=...` in the file), and
  `m.equilibrium()` returns the rightmost characteristic roots.
- **AUTO diagram points:** `p.multipliers` is what AUTO stores (exp(lambda)
  at equilibria, Floquet multipliers on periodic orbits); `p.eigenvalues`
  gives the eigenvalues of an equilibrium.
- **Two-parameter curves:** they are computed with tighter AUTO tolerances
  (1e-7) automatically.
- **One kernel, many models:** a notebook loads one model after another into
  the same XPP; each section builds and loads its own, so run a section from
  its start.
