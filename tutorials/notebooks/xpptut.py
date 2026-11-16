"""
Interactive helpers for the pyauto-xpp tutorials (VS Code / Jupyter).

Every number shown is computed by XPP 8.1 and its AUTO through pyxpp:
trajectories, nullclines, equilibria and eigenvalues, the direction field
(XPP's own right-hand side, or XPP's own direction-field routine), and the
bifurcation diagrams.

    from xpptut import Explorer
    ex = Explorer(mb, kind="plane", x="v", y="w", window=(-2.6, 2.6, -1, 2),
                  sliders={"i": (0, 1.6)}, auto="i")
    ex

kinds
    plane   phase plane: direction field (menu), nullclines, equilibria,
            click to start trajectories, a time plot, optional AUTO panel
    map     iterates of a map in a plane; click to start an orbit
    series  variables against time (and an optional pair plot)
    3d      a 3D trajectory, a projection and a time plot
    array   a chain of cells: space-time picture and selected cells
"""
from __future__ import annotations

import html as _html
import math

import ipywidgets as W
import numpy as np
import plotly.graph_objects as go
import pyxpp

DF_MODES = [
    "XPP direct field (length ~ speed)",
    "XPP scaled field (equal length)",
    "arrows coloured by speed",
    "arrows, length ~ speed^0.3",
    "flow (streamlines)",
    "speed map",
    "speed map + arrows",
    "speed map + flow",
    "none",
]
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d", "#4d7c0f"]


def load(mb, folder="odes"):
    """write the model's .ode file into `folder` (next to the notebooks) and load it"""
    from pathlib import Path
    return mb.load(Path(folder) / f"{mb.name}.ode")


def _white(fig, **kw):
    kw.setdefault("margin", dict(l=50, r=10, t=10, b=40))
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", **kw)
    return fig


def _break_wraps(a, period):
    """insert NaN where a folded variable jumps by more than half a period"""
    if period is None or len(a) < 2:
        return a
    a = np.asarray(a, float).copy()
    a[1:][np.abs(np.diff(a)) > period / 2] = np.nan
    return a


def _crossings(a, b):
    out = []
    for i in range(len(a) - 1):
        p, q = a[i], a[i + 1]
        if not (np.isfinite(p).all() and np.isfinite(q).all()):
            continue
        for k in range(len(b) - 1):
            r, s = b[k], b[k + 1]
            if not (np.isfinite(r).all() and np.isfinite(s).all()):
                continue
            d = (q[0] - p[0]) * (s[1] - r[1]) - (q[1] - p[1]) * (s[0] - r[0])
            if d == 0:
                continue
            t = ((r[0] - p[0]) * (s[1] - r[1]) - (r[1] - p[1]) * (s[0] - r[0])) / d
            u = ((r[0] - p[0]) * (q[1] - p[1]) - (r[1] - p[1]) * (q[0] - p[0])) / d
            if 0 <= t <= 1 and 0 <= u <= 1:
                out.append(p + t * (q - p))
    return out


def arrows(X, Y, U, V, window, length=0.022, power=None, scaled=True):
    """line segments (with heads) for arrows; lengths in screen units"""
    wx, wy = window[1] - window[0], window[3] - window[2]
    nu, nv = U / wx, V / wy
    L = np.hypot(nu, nv) + 1e-300
    if scaled:
        f = np.ones_like(L)
    elif power is not None:
        f = (L / L.max()) ** power
    else:
        f = L / L.max()
    sx, sy = [], []
    for x, y, a, b, s in zip(X.ravel(), Y.ravel(), (nu / L).ravel(), (nv / L).ravel(), f.ravel()):
        ux, uy = length * s * a, length * s * b
        tx, ty = x + ux * wx, y + uy * wy
        sx += [x - ux * wx, tx, None]
        sy += [y - uy * wy, ty, None]
        for sg in (1, -1):
            sx += [tx, tx + (-0.5 * ux - sg * 0.3 * uy) * wx, None]
            sy += [ty, ty + (-0.5 * uy + sg * 0.3 * ux) * wy, None]
    return sx, sy


def streamlines(X, Y, U, V, window, n_seeds=14, steps=120):
    """streamlines of the (XPP-computed) field, traced in screen units by RK2 on the grid"""
    xs, ys = X[0], Y[:, 0]
    wx, wy = window[1] - window[0], window[3] - window[2]
    nu, nv = U / wx, V / wy
    L = np.hypot(nu, nv) + 1e-300
    du, dv = nu / L, nv / L

    def field(px, py):
        i = np.clip((px - xs[0]) / (xs[-1] - xs[0]) * (len(xs) - 1), 0, len(xs) - 1.001)
        j = np.clip((py - ys[0]) / (ys[-1] - ys[0]) * (len(ys) - 1), 0, len(ys) - 1.001)
        i0, j0 = int(i), int(j)
        fi, fj = i - i0, j - j0
        a = (du[j0, i0] * (1 - fi) * (1 - fj) + du[j0, i0 + 1] * fi * (1 - fj)
             + du[j0 + 1, i0] * (1 - fi) * fj + du[j0 + 1, i0 + 1] * fi * fj)
        b = (dv[j0, i0] * (1 - fi) * (1 - fj) + dv[j0, i0 + 1] * fi * (1 - fj)
             + dv[j0 + 1, i0] * (1 - fi) * fj + dv[j0 + 1, i0 + 1] * fi * fj)
        return a, b

    h = 0.008
    sx, sy, hx, hy = [], [], [], []
    seeds = [(window[0] + (i + 0.5) / n_seeds * wx, window[2] + (j + 0.5) / n_seeds * wy)
             for i in range(n_seeds) for j in range(n_seeds)]
    for x0, y0 in seeds:
        line = [(x0, y0)]
        x, y = x0, y0
        for _ in range(steps):
            a, b = field(x, y)
            xm, ym = x + 0.5 * h * a * wx, y + 0.5 * h * b * wy
            a, b = field(xm, ym)
            x, y = x + h * a * wx, y + h * b * wy
            if not (window[0] <= x <= window[1] and window[2] <= y <= window[3]):
                break
            line.append((x, y))
        if len(line) > 4:
            px, py = zip(*line)
            sx += list(px) + [None]
            sy += list(py) + [None]
            k = len(line) // 2
            a, b = field(*line[k])
            ux, uy = 0.012 * a, 0.012 * b
            tx, ty = line[k]
            for sg in (1, -1):
                hx += [tx, tx + (-ux - sg * 0.6 * uy) * wx, None]
                hy += [ty, ty + (-uy + sg * 0.6 * ux) * wy, None]
    return sx + hx, sy + hy


class Explorer(W.VBox):
    """Interactive XPP explorer for one model built with ModelBuilder."""

    def __init__(self, mb, kind="plane", x=None, y=None, z=None, window=None, sliders=None,
                 auto=None, auto_range=None, auto2=None, auto2_range=None, auto_ds=None, auto_dsmax=None,
                 df=True, nullclines=True,
                 fold=None, total=None, show=None, cells=None, title=None, height=420):
        self.mb = mb
        self.m = load(mb)
        self.kind = kind
        names = [v.lower() for v in self.m.vars]
        self.x = (x or names[0]).lower()
        self.y = (y or (names[1] if len(names) > 1 else names[0])).lower()
        self.z = (z or (names[2] if len(names) > 2 else None))
        self.window = tuple(window) if window else (-2, 2, -2, 2)
        self.fold = {k.lower(): v for k, v in (fold or {}).items()}
        self.show_vars = [s.lower() for s in (show or self.m.vars[:4])]
        self.cells = cells
        self.starts = []
        self.eqs = []
        self.parts = []
        self.start_eq = None
        self.picked = None
        self.height = height
        self.df_ok = df and kind == "plane"
        self.nc_ok = nullclines and kind == "plane"

        # -- controls
        self.sliders = {}
        for p, rng in (sliders or {}).items():
            v = self.m.pars[p]
            lo, hi = rng
            lo, hi = min(lo, v), max(hi, v)
            s = W.FloatSlider(value=v, min=lo, max=hi, step=(hi - lo) / 500, description=p,
                              continuous_update=False, readout_format=".4g",
                              layout=W.Layout(width="330px"), style={"description_width": "70px"})
            s.observe(lambda ch: self.refresh(), names="value")
            self.sliders[p] = s
        num = self.m.numerics
        self.total = W.BoundedFloatText(value=total or num["total"], min=1e-6, max=1e9, description="total",
                                        layout=W.Layout(width="170px"))
        self.total.observe(lambda ch: self.refresh(), names="value")
        controls = [*self.sliders.values(), self.total]
        if self.df_ok:
            self.df_mode = W.Dropdown(options=DF_MODES, value=DF_MODES[1], description="field",
                                      layout=W.Layout(width="330px"))
            self.grid = W.IntSlider(value=16, min=6, max=40, description="grid", continuous_update=False,
                                    layout=W.Layout(width="330px"))
            self.arrow = W.FloatSlider(value=1.0, min=0.3, max=3, step=0.1, description="arrow size",
                                       continuous_update=False, layout=W.Layout(width="330px"))
            for w in (self.df_mode, self.grid, self.arrow):
                w.observe(lambda ch: self.refresh(), names="value")
            controls += [W.HTML("<b>direction field</b> (computed by XPP)"), self.df_mode, self.grid, self.arrow]
        self.show_nc = W.Checkbox(value=self.nc_ok, description="nullclines", indent=False)
        self.show_eq = W.Checkbox(value=kind == "plane", description="equilibria", indent=False)
        for w in (self.show_nc, self.show_eq):
            w.observe(lambda ch: self.refresh(), names="value")
        if kind == "plane":
            controls += [W.HBox([self.show_nc, self.show_eq])]
        clear = W.Button(description="clear", button_style="warning", layout=W.Layout(width="90px"))
        clear.on_click(lambda b: self.clear())
        rerun = W.Button(description="run again", layout=W.Layout(width="100px"))
        rerun.on_click(lambda b: self.refresh())
        controls += [W.HBox([clear, rerun])]
        self.info = W.HTML()

        # -- figures
        if kind == "3d":
            self.main = _white(go.FigureWidget(), height=height, width=560)
        else:
            self.main = _white(go.FigureWidget(), height=height, width=560, clickmode="event",
                               xaxis=dict(range=self.window[:2], title=self.x),
                               yaxis=dict(range=self.window[2:], title=self.y))
            gx, gy = np.meshgrid(np.linspace(self.window[0], self.window[1], 70),
                                 np.linspace(self.window[2], self.window[3], 70))
            self.main.add_scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", hoverinfo="none",
                                  marker=dict(size=8, opacity=0), showlegend=False)
            self.main.data[0].on_click(self._click)
            if self.df_ok or self.nc_ok:
                self.main.layout.on_change(self._zoomed, "xaxis.range", "yaxis.range")
        self.ts = _white(go.FigureWidget(), height=230, width=560, xaxis_title="t" if kind != "map" else "n")
        figs = [self.main, self.ts]
        hint = {"plane": "click in the phase plane to start a trajectory; zoom and XPP recomputes the field",
                "map": "click to start an orbit of the map", "series": "", "3d": "", "array": ""}[kind]
        middle = W.VBox([W.HTML(f"<i>{hint}</i>") if hint else W.HTML(), *figs, self.info])

        # -- AUTO
        right = []
        self.has_auto = auto is not None
        self.auto_ds, self.auto_dsmax = auto_ds, auto_dsmax
        if self.has_auto:
            # AUTO follows at most 8 parameters: make sure the ones used here are among them
            first = [auto] + ([auto2] if auto2 else []) + list(sliders or {})
            names = list(dict.fromkeys(first + self.m.par_names))[:8]
            self.m.auto.parameters(*names)
        if self.has_auto:
            par = auto
            pr = auto_range or (self.m.pars[par] - 1, self.m.pars[par] + 1)
            self.apar = W.Dropdown(options=self.m.par_names, value=par, description="parameter",
                                   layout=W.Layout(width="230px"))
            self.pmin = W.FloatText(value=pr[0], description="from", layout=W.Layout(width="150px"))
            self.pmax = W.FloatText(value=pr[1], description="to", layout=W.Layout(width="150px"))
            self.start_from = W.Dropdown(description="start from", options=[], layout=W.Layout(width="330px"))
            cont = W.Button(description="continue equilibria", button_style="primary")
            cont.on_click(lambda b: self.run_auto())
            self.bif = _white(go.FigureWidget(), height=330, width=500)
            self.pick_label = W.HTML("<i>then click an HB or LP point</i>")
            self.po_btn = W.Button(description="periodic orbits from HB", disabled=True)
            self.po_btn.on_click(lambda b: self.periodic())
            others = [p for p in self.m.par_names if p != par] or [None]
            self.apar2 = W.Dropdown(options=others, value=auto2 if auto2 in others else others[0],
                                    description="2nd par", layout=W.Layout(width="210px"))
            p2 = auto2_range or (0, 2)
            self.p2min = W.FloatText(value=p2[0], description="from", layout=W.Layout(width="150px"))
            self.p2max = W.FloatText(value=p2[1], description="to", layout=W.Layout(width="150px"))
            self.two_btn = W.Button(description="two-parameter curve", disabled=True)
            self.two_btn.on_click(lambda b: self.two_parameter())
            self.bif2 = _white(go.FigureWidget(), height=260, width=500)
            self.auto_msg = W.HTML()
            right = [W.HTML("<b>AUTO</b> (XPP's built-in)"), self.apar, W.HBox([self.pmin, self.pmax]),
                     *([self.start_from] if kind == "plane" else []), cont,
                     self.bif, self.pick_label, self.po_btn, self.apar2, W.HBox([self.p2min, self.p2max]),
                     self.two_btn, self.bif2, self.auto_msg]
        head = W.HTML(f"<h3 style='margin:4px 0'>{_html.escape(title)}</h3>") if title else W.HTML()
        cols = [W.VBox(controls, layout=W.Layout(width="350px")), middle]
        if right:
            cols.append(W.VBox(right))
        super().__init__([head, W.HBox(cols, layout=W.Layout(flex_flow="row wrap", align_items="flex-start"))])
        self.refresh()

    # ------------------------------------------------------------------ helpers
    def _sync(self):
        self.m.pars.update({p: s.value for p, s in self.sliders.items()})

    def set(self, **pars):
        """move sliders from code"""
        for k, v in pars.items():
            s = self.sliders.get(k)
            if s is None:
                self.m.pars[k] = v
                continue
            s.min, s.max = min(s.min, v), max(s.max, v)
            s.value = v
        self.refresh()

    def add_start(self, x0, y0):
        self.starts.append((float(x0), float(y0)))
        self.refresh()

    def clear(self):
        self.starts = []
        self.refresh()

    def _click(self, trace, points, state):
        if points.xs:
            self.add_start(points.xs[0], points.ys[0])

    def _zoomed(self, layout, xr, yr):
        if xr is None or yr is None:
            return
        new = (xr[0], xr[1], yr[0], yr[1])
        if np.allclose(new, self.window, rtol=1e-6):
            return
        self.window = new
        self.refresh()

    def run(self, **ics):
        """one run from the given initial conditions (others: the model's)"""
        self._sync()
        return self.m.run(total=self.total.value, keep=False, trans=0, **ics)

    def _line(self, tr, a, b):
        return _break_wraps(tr[a], self.fold.get(a)), _break_wraps(tr[b], self.fold.get(b))

    # ------------------------------------------------------------------ drawing
    def refresh(self):
        self._sync()
        getattr(self, "_draw_" + self.kind.replace("3d", "three"))()

    def _field_traces(self):
        mode = self.df_mode.value
        if mode == "none":
            return []
        win = self.window
        n = self.grid.value
        out = []
        size = 0.022 * self.arrow.value * 16 / max(n, 6)
        if mode.startswith("XPP"):
            a = self.m.direction_field(self.x, self.y, *win, grid=n, scaled="scaled" in mode)
            sx, sy = [], []
            for x0, y0, x1, y1 in a:
                sx += [x0, x1, None]
                sy += [y0, y1, None]
                dx, dy = (x1 - x0) / (win[1] - win[0]), (y1 - y0) / (win[3] - win[2])
                L = math.hypot(dx, dy) or 1
                hl = min(0.35 * L, 0.012)
                for sg in (1, -1):
                    sx += [x1, x1 + (-dx / L * hl - sg * 0.5 * dy / L * hl) * (win[1] - win[0]), None]
                    sy += [y1, y1 + (-dy / L * hl + sg * 0.5 * dx / L * hl) * (win[3] - win[2]), None]
            out.append(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color="#6b7280", width=1),
                                  hoverinfo="skip", name="XPP's field"))
            return out
        fine = "map" in mode or "flow" in mode
        X, Y, U, V = self.m.vector_field(self.x, self.y, *win, grid=60 if fine else n)
        S = np.hypot(U / (win[1] - win[0]), V / (win[3] - win[2]))
        if "map" in mode:
            out.append(go.Heatmap(x=X[0], y=Y[:, 0], z=np.log10(S + 1e-300), colorscale="Magma", opacity=0.55,
                                  showscale=True, colorbar=dict(title="log10 speed", len=0.6, thickness=10),
                                  hoverinfo="skip"))
        if "flow" in mode:
            sx, sy = streamlines(X, Y, U, V, win)
            out.append(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color="#374151", width=1),
                                  hoverinfo="skip", name="flow"))
        if "arrows" in mode:
            Xa, Ya, Ua, Va = (X, Y, U, V) if not fine else self.m.vector_field(self.x, self.y, *win, grid=n)
            Sa = np.hypot(Ua / (win[1] - win[0]), Va / (win[3] - win[2]))
            if "coloured" in mode:
                q = np.log10(Sa + 1e-300)
                edges = np.linspace(q.min(), q.max() + 1e-12, 7)
                cols = ["#440154", "#3b528b", "#21918c", "#5ec962", "#c8e020", "#fde725"]
                for k in range(6):
                    sel = (q >= edges[k]) & (q < edges[k + 1])
                    if sel.any():
                        sx, sy = arrows(Xa[sel], Ya[sel], Ua[sel], Va[sel], win, length=size)
                        out.append(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=cols[k], width=1.3),
                                              hoverinfo="skip", showlegend=False))
            else:
                power = 0.3 if "0.3" in mode else None
                sx, sy = arrows(Xa, Ya, Ua, Va, win, length=size, power=power, scaled=power is None)
                out.append(go.Scatter(x=sx, y=sy, mode="lines", hoverinfo="skip", showlegend=False,
                                      line=dict(color="white" if "map" in mode else "#4b5563", width=1)))
        return out

    def _equilibria(self, nc):
        found = []
        a = nc[self.x][np.isfinite(nc[self.x]).all(1)] if len(nc[self.x]) else np.empty((0, 2))
        b = nc[self.y][np.isfinite(nc[self.y]).all(1)] if len(nc[self.y]) else np.empty((0, 2))
        guesses = _crossings(nc[self.x], nc[self.y])
        sx = (self.window[1] - self.window[0])
        sy = (self.window[3] - self.window[2])
        if len(a) and len(b):
            d = np.hypot((a[:, None, 0] - b[None, :, 0]) / sx, (a[:, None, 1] - b[None, :, 1]) / sy)
            for i, j in zip(*np.where(d < 0.02)):
                if d[i, j] <= d[i].min() + 1e-12:
                    guesses.append((a[i] + b[j]) / 2)
        for g in guesses:
            try:
                e = self.m.equilibrium({self.x: g[0], self.y: g[1]})
            except pyxpp.XPPError:
                continue
            x, y = e.x[self.x.upper()], e.x[self.y.upper()]
            if not (self.window[0] <= x <= self.window[1] and self.window[2] <= y <= self.window[3]):
                continue
            if all(abs(x - f["x"]) / sx + abs(y - f["y"]) / sy > 1e-3 for f in found):
                re = e.eigenvalues.real
                kind = ("saddle" if (re > 0).any() and (re < 0).any() else
                        ("stable " if e.stable else "unstable ") +
                        ("focus" if np.any(np.abs(e.eigenvalues.imag) > 1e-9) else "node"))
                found.append(dict(x=x, y=y, stable=e.stable, kind=kind, state=dict(e.x),
                                  eig=", ".join(f"{z.real:.4g}{z.imag:+.4g}i" for z in e.eigenvalues)))
        return found

    def _draw_plane(self):
        traces = []
        if self.df_ok:
            traces += self._field_traces()
        nc = None
        if self.nc_ok and (self.show_nc.value or self.show_eq.value):
            nc = {k.lower(): v for k, v in self.m.nullclines(x=self.x, y=self.y, xlo=self.window[0],
                                                             xhi=self.window[1], ylo=self.window[2],
                                                             yhi=self.window[3], mesh=120).items()}
            if self.show_nc.value:
                for k, col in ((self.x, "#ea580c"), (self.y, "#16a34a")):
                    traces.append(go.Scatter(x=nc[k][:, 0], y=nc[k][:, 1], mode="lines", hoverinfo="skip",
                                             line=dict(color=col, width=2), name=f"{k}' = 0"))
        self.eqs = self._equilibria(nc) if (nc is not None and self.show_eq.value) else []
        for e in self.eqs:
            traces.append(go.Scatter(x=[e["x"]], y=[e["y"]], mode="markers", showlegend=False,
                                     marker=dict(size=12, color="black" if e["stable"] else "white",
                                                 line=dict(color="black", width=2),
                                                 symbol="x" if e["kind"] == "saddle" else "circle"),
                                     hovertext=f"{e['kind']}<br>{e['eig']}", hoverinfo="text"))
        last = None
        for k, (x0, y0) in enumerate(self.starts):
            tr = self.run(**{self.x: x0, self.y: y0})
            lx, ly = self._line(tr, self.x, self.y)
            traces.append(go.Scatter(x=lx, y=ly, mode="lines", line=dict(width=1.7, color=COLORS[k % 8]),
                                     hoverinfo="skip", name=f"from ({x0:.3g}, {y0:.3g})"))
            last = tr
        if last is None:
            last = self.run()
            lx, ly = self._line(last, self.x, self.y)
            traces.append(go.Scatter(x=lx, y=ly, mode="lines", line=dict(width=1.5, color="#111827"),
                                     hoverinfo="skip", name="from the file's initial conditions"))
        self._update_main(traces)
        self._time_plot(last)
        if self.has_auto:
            opts = [(f"{e['kind']} at ({e['x']:.4g}, {e['y']:.4g})", i) for i, e in enumerate(self.eqs)]
            stable = [i for i, e in enumerate(self.eqs) if e["stable"]]
            old = self.start_from.value
            self.start_from.options = opts
            if opts:
                self.start_from.value = old if old is not None and old < len(opts) else (stable[0] if stable else 0)
        rows = "".join(f"<tr><td>{e['x']:.5g}</td><td>{e['y']:.5g}</td><td>{e['kind']}</td><td>{e['eig']}</td></tr>"
                       for e in self.eqs)
        self.info.value = ("<table style='font-family:monospace;font-size:12px'><tr><th>" + self.x + "</th><th>"
                           + self.y + "</th><th>kind</th><th>eigenvalues</th></tr>" + rows + "</table>") if rows else ""

    def _draw_map(self):
        traces = []
        starts = self.starts or [(None, None)]
        last = None
        for k, (x0, y0) in enumerate(starts):
            ics = {} if x0 is None else {self.x: x0, self.y: y0}
            tr = self.run(**ics)
            traces.append(go.Scattergl(x=tr[self.x], y=tr[self.y], mode="markers", hoverinfo="skip",
                                       marker=dict(size=2.5, color=COLORS[k % 8]), name=f"orbit {k + 1}"))
            last = tr
        self._update_main(traces)
        self._time_plot(last, mode="markers", limit=200)

    def _draw_series(self):
        tr = self.run()
        traces = []
        if self.x != self.y:
            lx, ly = self._line(tr, self.x, self.y)
            traces.append(go.Scatter(x=lx, y=ly, mode="lines", line=dict(width=1.4, color="#111827"),
                                     hoverinfo="skip", showlegend=False))
        self._update_main(traces)
        self._time_plot(tr)

    def _draw_three(self):
        tr = self.run()
        with self.main.batch_update():
            self.main.data = []
            self.main.add_scatter3d(x=tr[self.x], y=tr[self.y], z=tr[self.z.lower()], mode="lines",
                                    line=dict(width=2, color=tr.t, colorscale="Viridis"))
            self.main.update_layout(scene=dict(xaxis_title=self.x, yaxis_title=self.y, zaxis_title=self.z))
        self._time_plot(tr)

    def _draw_array(self):
        tr = self.run()
        cols = [c for c in tr.names if c.lower().startswith(self.cells[0])]
        cols.sort(key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or 0))
        Z = np.array([tr[c] for c in cols])
        with self.main.batch_update():
            self.main.data = self.main.data[:1]
            self.main.add_heatmap(x=tr.t, y=list(range(1, len(cols) + 1)), z=Z, colorscale="RdBu_r",
                                  colorbar=dict(title=self.cells[0], len=0.7))
            self.main.update_layout(xaxis=dict(title="t", range=[tr.t[0], tr.t[-1]]),
                                    yaxis=dict(title="cell", range=[0.5, len(cols) + 0.5]))
        with self.ts.batch_update():
            self.ts.data = []
            for k, c in enumerate(self.cells[1]):
                self.ts.add_scatter(x=tr.t, y=tr[c], name=c, line=dict(color=COLORS[k % 8]))

    def _update_main(self, traces):
        with self.main.batch_update():
            self.main.data = self.main.data[:1]
            self.main.add_traces(traces)
            self.main.layout.xaxis.range = self.window[:2]
            self.main.layout.yaxis.range = self.window[2:]

    def _time_plot(self, tr, mode="lines", limit=None):
        with self.ts.batch_update():
            self.ts.data = []
            if tr is None:
                return
            sl = slice(0, limit) if limit else slice(None)
            for k, v in enumerate(self.show_vars):
                self.ts.add_scatter(x=tr.t[sl], y=tr[v][sl], name=v, mode=mode,
                                    marker=dict(size=3), line=dict(width=1.3, color=COLORS[k % 8]))

    # ------------------------------------------------------------------ AUTO
    def _start_point(self):
        """an equilibrium to start AUTO from: the stable one in the window, or the end of a long run"""
        self.m.buttons.numerics.sing_pt_ctrl(500, 1e-11, 1e-7)
        if self.eqs:                     # the equilibrium chosen in the menu (phase planes)
            e = self.eqs[self.start_from.value or 0]
            state = {k.lower(): v for k, v in e["state"].items()}
            if e["stable"]:              # settle on it by integrating
                tr = self.m.run(total=max(self.total.value, 500), keep=False, trans=0, **state)
                return {v: float(tr[v][-1]) for v in self.m.vars}
            return dict(self.m.equilibrium(state).x)
        # a run that settles gives the most accurate start
        tr = self.m.run(total=max(self.total.value, 500), keep=False, trans=0)
        late = tr.t > tr.t[-1] * 0.8 if len(tr) > 10 else np.zeros(len(tr), bool)
        if late.sum() > 5 and all(np.ptp(tr[v][late]) < 1e-6 * max(1.0, np.abs(tr[v][late]).max()) for v in self.m.vars):
            return {v: float(tr[v][-1]) for v in self.m.vars}
        # otherwise Newton from the end of the run, with a tight tolerance
        # (XPP's default, 1e-3, is too rough to start AUTO)
        return dict(self.m.equilibrium({v: float(tr[v][-1]) for v in self.m.vars}).x)

    def _steps(self, span):
        """AUTO step sizes: small, because XPP's AUTO can loop for ever on sharp folds
        with a large maximum step (seen in Wilson-Cowan with dsmax = 0.05)"""
        return (self.auto_ds or 0.001 * span), (self.auto_dsmax or 0.004 * span)

    def run_auto(self, start_from=None):
        """AUTO: equilibria in the chosen parameter, both ways from an equilibrium
        (start_from: index into the equilibria of the phase plane)"""
        if start_from is not None:
            self.start_from.value = start_from
        self._sync()
        keep = self.m.pars.as_dict()
        start = self._start_point()
        self.start_eq = start
        par, lo, hi = self.apar.value, self.pmin.value, self.pmax.value
        ds, dsmax = self._steps(hi - lo)
        parts = []
        for sign in (1, -1):
            self.m.auto.reset()
            self.m.pars.update(keep)
            self.m.set(**{k.lower(): v for k, v in start.items()})
            self.m.run(total=1e-4, keep=False)
            d = self.m.auto.run(par, start="steady", ds=sign * ds, dsmax=dsmax, dsmin=1e-7,
                                parmin=lo, parmax=hi, nmax=4000, npr=4000)
            parts.append(d)
        self.m.pars.update(keep)
        self.parts = [("eq", d) for d in parts]
        self._draw_bif()
        sp = sorted({(p.type, round(p.x, 6)) for d in parts for p in d.labeled() if p.type in ("HB", "LP", "BP")},
                    key=lambda t: t[1])
        self.auto_msg.value = ("<pre style='font-size:12px'>" + _html.escape(
            "\n".join(f"{t} at {par} = {x}" for t, x in sp) or "no special points") + "</pre>")
        return parts

    def _draw_bif(self):
        y = self.x
        traces = []
        special = []
        for kind, d in self.parts:
            pts = list(d)
            if not pts:
                continue
            xs = np.array([p.x for p in pts])
            st = np.array([p.stable for p in pts])
            if kind == "po":
                for s, col in ((True, "#16a34a"), (False, "#2563eb")):
                    for f in ("umax", "umin"):
                        traces.append(go.Scatter(x=xs[st == s], y=np.array([getattr(p, f)[y] for p in pts])[st == s],
                                                 mode="markers", showlegend=False, hoverinfo="skip",
                                                 marker=dict(size=5, color=col if s else "white",
                                                             line=dict(color=col, width=1))))
            else:
                yy = np.array([p.u0[y] for p in pts])
                for s, col, w in ((True, "#dc2626", 3), (False, "black", 1.2)):
                    traces.append(go.Scatter(x=xs, y=np.where(st == s, yy, np.nan), mode="lines",
                                             showlegend=False, hoverinfo="skip", line=dict(color=col, width=w)))
                special += [p for p in pts if p.type in ("HB", "LP")]
        with self.bif.batch_update():
            self.bif.data = []
            self.bif.add_traces(traces)
            self.bif.add_scatter(x=[p.x for p in special], y=[p.u0[y] for p in special], mode="markers+text",
                                 text=[p.type for p in special], textposition="top right",
                                 customdata=[p.type for p in special], marker=dict(size=12, color="#7c3aed"),
                                 hovertext=[f"{p.type} at {p.x:.6g}" for p in special], hoverinfo="text",
                                 showlegend=False)
            self.bif.update_layout(xaxis_title=self.apar.value, yaxis_title=y)
        self.bif.data[-1].on_click(self._bif_click)

    def _bif_click(self, trace, points, state):
        if points.point_inds:
            i = points.point_inds[0]
            self.pick(trace.customdata[i], trace.x[i])

    def pick(self, kind, x):
        self.picked = (kind, x)
        self.pick_label.value = f"picked <b>{kind}</b> at {self.apar.value} = {x:.6g}"
        self.po_btn.disabled = kind != "HB"
        self.two_btn.disabled = False

    def _refind(self, kind, x):
        keep = self.m.pars.as_dict()
        par, lo, hi = self.apar.value, self.pmin.value, self.pmax.value
        ds, dsmax = self._steps(hi - lo)
        sign = 1 if x >= keep[par] else -1
        self.m.auto.reset()
        self.m.set(**{k.lower(): v for k, v in self.start_eq.items()})
        self.m.run(total=1e-4, keep=False)
        d = self.m.auto.run(par, start="steady", ds=sign * ds, dsmax=dsmax, parmin=lo, parmax=hi, nmax=4000)
        pts = d.special(kind)
        return (min(pts, key=lambda p: abs(p.x - x)) if pts else None), keep, ds, dsmax

    def periodic(self):
        """AUTO: periodic orbits from the picked Hopf point; returns (Diagram, l1)"""
        kind, x = self.picked
        p, keep, ds, dsmax = self._refind(kind, x)
        if p is None:
            self.auto_msg.value = "Hopf point not found again"
            return None, None
        l1 = self.m.auto.lyapunov(p)[0]
        po = self.m.auto.run(grab=p.label, choice="periodic", ntst=60, ds=ds, dsmax=dsmax, nmax=600)
        self.m.pars.update(keep)
        self.parts = [q for q in self.parts if q[0] == "eq"] + [("po", po)]
        self._draw_bif()
        how = ("degenerate (l1 = 0)" if abs(l1) < 1e-6 else
               "subcritical: unstable orbits are born" if l1 > 0 else "supercritical: stable orbits are born")
        lps = sorted({round(q.x, 5) for q in po.special("LP")})
        self.auto_msg.value = (f"<pre style='font-size:12px'>Hopf at {self.apar.value} = {p.x:.6g}: "
                               f"l1 = {l1:.4g}, {how}\n{len(po)} orbits"
                               + (f"; folds of orbits at {lps}" if lps else "") + "</pre>")
        return po, l1

    def two_parameter(self):
        """AUTO: the picked point in two parameters (with GH points on Hopf curves)"""
        kind, x = self.picked
        par2 = self.apar2.value
        p, keep, ds, _ = self._refind(kind, x)
        if p is None or par2 is None:
            self.auto_msg.value = "point not found again (or no second parameter)"
            return [], []
        lo = min(self.pmin.value, self.p2min.value)
        hi = max(self.pmax.value, self.p2max.value)
        # one step for both parameters: size it on the smaller range
        ds, dsmax = self._steps(min(self.pmax.value - self.pmin.value, self.p2max.value - self.p2min.value))
        num = dict(param2=par2, nmax=2000, dsmax=dsmax, parmin=lo, parmax=hi)
        ch = dict(choice="two param") if kind == "HB" else {}
        c1 = self.m.auto.run(grab=p.label, ds=ds, **ch, **num)
        c2 = self.m.auto.run(grab=p.label, ds=-ds, **ch)
        gh = (self.m.auto.gh_points(c1) + self.m.auto.gh_points(c2)) if kind == "HB" else []
        self.m.pars.update(keep)
        with self.bif2.batch_update():
            self.bif2.data = []
            for c in (c1, c2):
                self.bif2.add_scatter(x=[q.x for q in c], y=[q.par[par2] for q in c], mode="lines",
                                      line=dict(color="#dc2626" if kind == "HB" else "#2563eb"), showlegend=False)
            for g in gh:
                self.bif2.add_scatter(x=[g["par"]], y=[g["par2"]], mode="markers+text", text=["GH"],
                                      textposition="top right", marker=dict(size=10, color="#7c3aed"),
                                      showlegend=False)
            self.bif2.update_layout(xaxis_title=self.apar.value, yaxis_title=par2,
                                    xaxis_range=[self.pmin.value, self.pmax.value],
                                    yaxis_range=[self.p2min.value, self.p2max.value])
        self.auto_msg.value = (f"<pre style='font-size:12px'>{kind} curve in ({self.apar.value}, {par2}): "
                               f"{len(c1) + len(c2)} points"
                               + "".join(f"\nGH at {g['par']:.5g}, {g['par2']:.5g}" for g in gh) + "</pre>")
        return [c1, c2], gh


def orbit_diagram(mb, par, values, var, transient=500, keep=200, ics=None):
    """iterate a map at each parameter value; returns (par values, states) for a scatter plot"""
    m = load(mb)
    P, X = [], []
    for v in values:
        tr = m.run(keep=False, total=transient + keep, trans=transient, **{par: v}, **(ics or {}))
        P += [v] * len(tr)
        X += list(tr[var])
    return np.array(P), np.array(X)


def show_figure(fig, height=380, **layout):
    """a plain (non-widget) Plotly figure with the tutorial style"""
    _white(fig, height=height, **layout)
    fig.show()
    return fig
