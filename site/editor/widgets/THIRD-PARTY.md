# Third-party code in the editor's widget runtime (widgets.js, widgets.css, fonts)

The bundle is built from these npm packages (exact versions in package-lock.json):

| package | licence |
|---|---|
| @jupyter-widgets/html-manager, base, base-manager, controls, output (Project Jupyter, ipywidgets 8.1) | BSD-3-Clause |
| @lumino/* (Project Jupyter) | BSD-3-Clause |
| @jupyterlab/* (rendermime, services, ...) | BSD-3-Clause |
| anywidget (Trevor Manz) | MIT |
| Font Awesome Free 5 (icons used by ipywidgets) | fonts: SIL OFL 1.1; CSS: MIT |
| jquery, backbone, underscore, d3-format, nouislider, and other dependencies | MIT / BSD |

The licence texts are in each package (node_modules/<package>/LICENSE) of a build;
`npm ls --all` in editor-widgets/ lists every package and version.
