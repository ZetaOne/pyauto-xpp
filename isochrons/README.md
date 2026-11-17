# Isochrons

Notebooks about isochrons: the curves of equal asymptotic phase of an oscillator. They start
from the idea itself and lead to the global isochrons of the reduced Hodgkin-Huxley neuron
computed by continuation, as in H. M. Osinga and J. Moehlis, *Continuation-based computation
of global isochrons*, SIAM J. Appl. Dyn. Syst. 9 (2010). Everything is computed with XPPAUT 8.2
and its AUTO through pyauto-xpp.

| # | notebook | what it shows |
|---|---|---|
| 01 | [01_FHN_basics-isochron.ipynb](01_FHN_basics-isochron.ipynb) ([Open in Colab](https://colab.research.google.com/github/ZetaOne/pyauto-xpp/blob/main/isochrons/01_FHN_basics-isochron.ipynb)) | What an isochron is, with FitzHugh-Nagumo: phase on the cycle, asymptotic phase, the phase map and its contour lines, points on one isochron moving together, kicks and the phase response curve; then the reduced Hodgkin-Huxley neuron and why its isochrons are hard |

The animations in `gifs/` are shown in the notebooks. Run locally with
`pixi run isochrons` (JupyterLab on this folder), or open a notebook in Colab: its first cell
installs pyauto-xpp.
