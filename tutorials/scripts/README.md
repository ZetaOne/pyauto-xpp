# py-tuts: pyauto-xpp by scripting

Plain Python scripts, one per model, that use XPPAUT 8.2 and its AUTO through
pyauto-xpp (`pyxpp`). No widgets: each script builds its model in code with
`ModelBuilder`, simulates, analyses, saves matplotlib figures and checks its
numbers against exact or published values.

```
conda activate xpp-dev
python 2D-FLOWS_fitzhugh-nagumo.py      # one script
python run_all.py                        # all of them, one line each
python run_all.py MAPS DELAYS            # only some groups
```

Each script:

- prints the `.ode` file it builds (also written to `odes/`),
- saves its figures to `figures/<script name>/`,
- ends with `N/M checks passed`.

Scripts are split into cells with `# %%`, so in VS Code you can also step
through them with *Run Cell*.

The models come from two sources:

- the examples of XPP Studio (Peter Ashwin and Bard Ermentrout, GPL-3.0);
- the delay-equation examples that come with XPPAUT 8.2.

## 2D-FLOWS

| script | model | what it shows |
|---|---|---|
| `2D-FLOWS_fitzhugh-nagumo.py` | FitzHugh-Nagumo | phase plane, excitability, Hopf points (exact), periodic orbits |
| `2D-FLOWS_toggle-switch.py` | genetic toggle switch | bistability, basins, folds and the cusp in (a1, a2) |
| `2D-FLOWS_wilson-cowan.py` | Wilson-Cowan | S-shaped branch with folds, Hopf in the inhibitory time constant |
| `2D-FLOWS_morris-lecar.py` | Morris-Lecar (Hopf regime) | f-I curve, class II onset, hysteresis |
| `2D-FLOWS_morris-lecar-bifurcation.py` | Morris-Lecar | one- and two-parameter diagrams, GH points |
| `2D-FLOWS_morris-lecar-homoclinic.py` | Morris-Lecar with a saddle | period blow-up at a homoclinic orbit |
| `2D-FLOWS_hodgkin-huxley.py` | Hodgkin-Huxley | period, Hopf points, fold of orbits, bistability |
| `2D-FLOWS_interacting-saddle-nodes.py` | interacting saddle-nodes | equilibria from a quartic, folds, Hopf at mu = 0 |
| `2D-FLOWS_stiff-van-der-pol.py` | van der Pol, mu = 1000 | stiff solvers, relaxation-oscillation period |
| `2D-FLOWS_driven-pendulum.py` | driven pendulum | rest vs rotation, hysteresis, Melnikov onset |
| `2D-FLOWS_forced-duffing.py` | forced Duffing | stroboscopic map, period doubling to chaos |
| `2D-FLOWS_forced-phase-oscillator.py` | forced phase oscillator | devil's staircase, Arnold tongues |
| `2D-FLOWS_phase-oscillators-table.py` | phase oscillators with a table | locking band from the tabulated coupling |
| `2D-FLOWS_coupled-ginzburg-landau.py` | Ginzburg-Landau oscillator | adjoint, H and H2 against exact formulas |

## EVENTS

| script | model | what it shows |
|---|---|---|
| `EVENTS_bouncing-ball.py` | bouncing ball | event times, restitution, Zeno time |
| `EVENTS_integrate-and-fire.py` | integrate-and-fire with adaptation | exact intervals, f-I curves, rheobase |

## MAPS

| script | model | what it shows |
|---|---|---|
| `MAPS_logistic-map.py` | logistic map | orbit diagram, Liapunov exponent, period doubling (exact points) |
| `MAPS_delayed-logistic-map.py` | delayed logistic map | Neimark-Sacker at a = 2, invariant curve |
| `MAPS_henon-map.py` | Hénon map | strange attractor, both Liapunov exponents |
| `MAPS_chirikov-taylor-map.py` | standard map | invariant curves vs chaos, area preservation |
| `MAPS_piecewise-affine-map.py` | piecewise affine map | periods and islands at the golden mean |

## 3D-FLOWS

| script | model | what it shows |
|---|---|---|
| `3D-FLOWS_lorenz.py` | Lorenz | pitchfork and Hopf (exact), Liapunov exponent, Lorenz map |
| `3D-FLOWS_rossler.py` | Rössler | equilibria, period doubling in c |
| `3D-FLOWS_lorenz-84.py` | Lorenz 84 | folds and Hopf of the westerly flow, Hadley state |

## DELAYS

| script | model | what it shows |
|---|---|---|
| `DELAYS_delayed-logistic.py` | Hutchinson equation | Hopf at r tau = pi/2 from simulations |
| `DELAYS_mackey-glass.py` | Mackey-Glass | maxima against tau, delay embedding |
| `DELAYS_distributed-delay-volterra.py` | gamma-kernel Volterra equation | Volterra vs linear-chain ODE, onset at r tau = 2 |
| `DELAYS_hutchinson-periodic-orbits.py` | Hutchinson equation | AUTO on a delay equation: Hopf and periodic orbits (XPP 8.2) |
| `DELAYS_mackey-glass-period-doubling.py` | Mackey-Glass | period-doubling cascade with AUTO (XPP 8.2) |
| `DELAYS_cusp-delay.py` | cusp with a delay | folds and delay-induced Hopf points (XPP 8.2) |
| `DELAYS_marcus-westervelt.py` | two delay-coupled neurons | Hopf points and the (a, tau) Hopf curve (XPP 8.2) |
| `DELAYS_two-delays.py` | equation with two delays | Hopf points with two delays (XPP 8.2) |
| `DELAYS_wilson-cowan-delay.py` | Wilson-Cowan with delay | (P, tau) Hopf curve (XPP 8.2) |
| `DELAYS_stuart-landau-single.py` | delayed Stuart-Landau | rotating waves and their folds (XPP 8.2) |
| `DELAYS_stuart-landau-pair.py` | two delay-coupled Stuart-Landau | torus bifurcations of the in-phase wave (XPP 8.2) |

## NOISE

| script | model | what it shows |
|---|---|---|
| `NOISE_noisy-fhn-sde.py` | FitzHugh-Nagumo with white noise | coherence resonance (seeded, reproducible) |
| `NOISE_telegraph-fhn-markov.py` | FitzHugh-Nagumo with a Markov input | dwell times, fraction on, firing only when on |

## SPACE

| script | model | what it shows |
|---|---|---|
| `SPACE_fhn-chain-arrays.py` | chain of 20 FitzHugh-Nagumo cells | travelling spikes, speed ~ sqrt(d) |
| `SPACE_nagumo-front.py` | Nagumo front | front speed by shooting vs (1 - 2a)/sqrt(2) |

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
