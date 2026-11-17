# pyauto-xpp tutorials

Interactive notebooks built on the examples of [XPP Studio](https://xpp-studio.github.io/xpp-studio/) (Peter Ashwin and Bard Ermentrout,
GPL-3.0). Each notebook writes its model with `ModelBuilder` and explores it with XPPAUT 8.2
and its AUTO through pyauto-xpp.

Open a notebook in JupyterLab (`pixi run lab`) and run all cells. `xpptut.py` is the
shared interactive explorer (sliders, a direction-field menu, clickable phase planes, AUTO).

Work through them in number order: each level builds on the one before.


## Level 1 — First steps

Simulate, read a phase plane, find equilibria, handle events, iterate simple maps, meet chaos.

| # | notebook | topic |
|---|---|---|
| 01 | [01_fitzhugh_nagumo.ipynb](01_fitzhugh_nagumo.ipynb) | FitzHugh-Nagumo: an excitable membrane |
| 02 | [02_toggle_switch.ipynb](02_toggle_switch.ipynb) | Genetic toggle switch: bistability |
| 03 | [03_bouncing_ball.ipynb](03_bouncing_ball.ipynb) | Bouncing ball: global events |
| 04 | [04_integrate_and_fire.ipynb](04_integrate_and_fire.ipynb) | Integrate-and-fire neuron with adaptation |
| 05 | [05_logistic_map.ipynb](05_logistic_map.ipynb) | The logistic map: period doubling to chaos |
| 06 | [06_henon_map.ipynb](06_henon_map.ipynb) | The Hénon map: a strange attractor |
| 07 | [07_lorenz.ipynb](07_lorenz.ipynb) | The Lorenz equations |

## Level 2 — Bifurcations with AUTO

Follow equilibria and periodic orbits in one parameter: Hopf, folds, period doubling.

| # | notebook | topic |
|---|---|---|
| 08 | [08_wilson_cowan.ipynb](08_wilson_cowan.ipynb) | Wilson-Cowan: excitatory and inhibitory populations |
| 09 | [09_morris_lecar.ipynb](09_morris_lecar.ipynb) | Morris-Lecar: a neuron with a Hopf onset |
| 10 | [10_hodgkin_huxley.ipynb](10_hodgkin_huxley.ipynb) | Hodgkin-Huxley: the squid giant axon |
| 11 | [11_stiff_van_der_pol.ipynb](11_stiff_van_der_pol.ipynb) | Stiff van der Pol: when to use a stiff solver |
| 12 | [12_driven_pendulum.ipynb](12_driven_pendulum.ipynb) | Damped driven pendulum on a cylinder |
| 13 | [13_delayed_logistic_map.ipynb](13_delayed_logistic_map.ipynb) | Delayed logistic map: a Neimark-Sacker bifurcation |
| 14 | [14_rossler.ipynb](14_rossler.ipynb) | The Rössler system: period doubling in c |
| 15 | [15_lorenz_84.ipynb](15_lorenz_84.ipynb) | Lorenz 84: a model of the atmosphere |

## Level 3 — Two parameters and global bifurcations

Curves of Hopf and fold points; homoclinic and heteroclinic orbits.

| # | notebook | topic |
|---|---|---|
| 16 | [16_morris_lecar_bifurcation.ipynb](16_morris_lecar_bifurcation.ipynb) | Morris-Lecar: one- and two-parameter diagrams |
| 17 | [17_morris_lecar_homoclinic.ipynb](17_morris_lecar_homoclinic.ipynb) | Morris-Lecar with a saddle: a homoclinic orbit |
| 18 | [18_interacting_saddle_nodes.ipynb](18_interacting_saddle_nodes.ipynb) | Interacting saddle-nodes |
| 19 | [19_nagumo_front.ipynb](19_nagumo_front.ipynb) | Nagumo travelling front: a heteroclinic orbit |

## Level 4 — Forcing, tori and phase

Forced oscillators, maps on a torus, locking, phase reduction.

| # | notebook | topic |
|---|---|---|
| 20 | [20_forced_duffing.ipynb](20_forced_duffing.ipynb) | Periodically forced Duffing oscillator |
| 21 | [21_forced_phase_oscillator.ipynb](21_forced_phase_oscillator.ipynb) | Forced phase oscillator on a torus: Arnold tongues |
| 22 | [22_phase_oscillators_table.ipynb](22_phase_oscillators_table.ipynb) | Two phase oscillators with a tabulated coupling |
| 23 | [23_chirikov_taylor_map.ipynb](23_chirikov_taylor_map.ipynb) | Chirikov-Taylor (standard) map |
| 24 | [24_piecewise_affine_map.ipynb](24_piecewise_affine_map.ipynb) | A piecewise affine map on the unit square |
| 25 | [25_coupled_ginzburg_landau.ipynb](25_coupled_ginzburg_landau.ipynb) | Ginzburg-Landau oscillator: phase reduction |

## Level 5 — Noise and space

Stochastic and Markov inputs; chains of coupled cells.

| # | notebook | topic |
|---|---|---|
| 26 | [26_noisy_fhn_sde.ipynb](26_noisy_fhn_sde.ipynb) | Noisy FitzHugh-Nagumo: coherence from noise |
| 27 | [27_telegraph_fhn_markov.ipynb](27_telegraph_fhn_markov.ipynb) | FitzHugh-Nagumo with a random on/off input |
| 28 | [28_fhn_chain_arrays.ipynb](28_fhn_chain_arrays.ipynb) | A chain of FitzHugh-Nagumo cells: waves |

## Level 6 — Delay equations

From simulations of delay equations to AUTO on them (XPP 8.2).

| # | notebook | topic |
|---|---|---|
| 29 | [29_delayed_logistic.ipynb](29_delayed_logistic.ipynb) | Hutchinson's delayed logistic equation |
| 30 | [30_mackey_glass.ipynb](30_mackey_glass.ipynb) | Mackey-Glass: delay-induced chaos |
| 31 | [31_distributed_delay_volterra.ipynb](31_distributed_delay_volterra.ipynb) | Distributed delay as a Volterra integral |
