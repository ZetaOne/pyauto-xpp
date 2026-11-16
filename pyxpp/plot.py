"""
Plots of XPP's data, and XPP's own drawing of the same view as a check.

Rung R1 (integrate, x vs t):

    tr = m.run()                                    # integration done by XPP
    lim = window(tr, "v")                           # t range and y range with a margin
    fig, ax = plot_x_vs_t(tr, "v", lim)             # matplotlib
    svg = xpp_graph(m, "v", lim, "xpp_v.svg")       # XPP draws v vs t itself (headless)
    check = compare_with_xpp_graph(tr, "v", svg)    # the two curves agree to pixel rounding?

XPP's SVG has the curve as line segments in whole pixels (class xppline*).
compare_with_xpp_graph fits pixel = a*value + b on each axis to our data
and reports the largest distance in pixels; rounding alone gives at most
about 0.5-1 pixel.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def window(tr, var, pad=0.05):
    """(tlo, thi, ylo, yhi): the time span of the run and the range of `var`
    with a margin (a flat curve gets +-1)."""
    t, y = tr.t, tr[var]
    fin = np.isfinite(y)
    lo, hi = (float(np.min(y[fin])), float(np.max(y[fin]))) if fin.any() else (-1.0, 1.0)
    if hi - lo < 1e-12 * max(1.0, abs(hi)):
        lo, hi = lo - 1, hi + 1
    m = pad * (hi - lo)
    tlo, thi = float(t[0]), float(t[-1])
    if thi == tlo:
        thi = tlo + 1
    return tlo, thi, lo - m, hi + m


def plot_x_vs_t(tr, var, lim=None, ax=None, title=None, **style):
    """`var` against t from XPP's data."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4))
    else:
        fig = ax.figure
    style = {"color": "k", "lw": 1, **style}
    ax.plot(tr.t, tr[var], **style)
    lim = lim or window(tr, var)
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set(xlabel="t", ylabel=var)
    if title:
        ax.set_title(title)
    return fig, ax


def xpp_graph(m, var, lim, path):
    """XPP's own graph of `var` against t in the window `lim`, as SVG
    (Viewaxes > 2D, Restore, Graphic stuff > SVG).  Nothing is integrated
    again, so a noisy model shows the same run."""
    tlo, thi, ylo, yhi = lim
    m.view(x="t", y=var, xlo=tlo, xhi=thi, ylo=ylo, yhi=yhi)
    m.buttons.graphic_stuff.remove_all()   # only the main curve (a file may add more with nplot)
    m.press("r")                       # Restore: redraw from the stored data
    path = Path(path)
    rel = path.resolve()
    m.buttons.graphic_stuff.svg(str(rel))
    if not path.exists():
        raise RuntimeError(f"XPP did not write {path}")
    return path


def svg_curve(path):
    """The points of the curves XPP drew (classes xppline*), in pixels,
    as an (N, 2) array; NaN rows separate disconnected pieces."""
    text = Path(path).read_text(errors="replace")
    pts = []
    last = None
    for m in re.finditer(r'<line class="xppline\d+"\s+x1="([-\d.]+)"\s+y1="([-\d.]+)"\s+'
                         r'x2="([-\d.]+)"\s+y2="([-\d.]+)"', text):
        # one segment per stored row, from the previous row to this one
        # (the first segment goes from row 0 to row 0): the end points are
        # the rows
        x1, y1, x2, y2 = map(float, m.groups())
        if last is not None and (x1, y1) != last:
            pts.append((np.nan, np.nan))
        pts.append((x2, y2))
        last = (x2, y2)
    return np.array(pts, dtype=float).reshape(-1, 2)


def compare_with_xpp_graph(tr, var, svg_path, lim=None):
    """Does XPP's drawn curve match our data?  Returns a dict with the
    number of points, the largest misfit in pixels, and ok (<= 1.5 px).

    XPP draws one segment per stored row inside the window; the points are
    matched in order to the rows of the data."""
    px = svg_curve(svg_path)
    px = px[np.isfinite(px).all(axis=1)]
    t, y = np.asarray(tr.t, float), np.asarray(tr[var], float)
    lim = lim or window(tr, var)
    inside = (t >= lim[0]) & (t <= lim[1]) & np.isfinite(y)
    t, y = t[inside], y[inside]
    out = {"xpp_points": int(len(px)), "data_points": int(len(t))}
    n = min(len(px), len(t))
    if n < 3:
        out.update(ok=False, why="too few points")
        return out
    # XPP drops repeated pixels only through identical consecutive segments;
    # use the first n points of each
    P, T, Y = px[:n], t[:n], y[:n]
    res = []
    for data, pix in ((T, P[:, 0]), (Y, P[:, 1])):
        A = np.vstack([data, np.ones_like(data)]).T
        coef, *_ = np.linalg.lstsq(A, pix, rcond=None)
        res.append(np.abs(A @ coef - pix))
    err = np.maximum(res[0], res[1])
    out.update(max_px=float(err.max()), mean_px=float(err.mean()),
               ok=bool(err.max() <= 1.5 and len(px) == len(t)))
    if len(px) != len(t):
        out["why"] = "XPP drew a different number of points"
    return out


# ---------------------------------------------------------------------------
# Rung R2: all variables against t, x_i vs x_j, 3D
# ---------------------------------------------------------------------------
def span(tr, var, pad=0.05):
    """(lo, hi) of a column with a margin (a flat column gets +-1)."""
    y = np.asarray(tr.t if var.lower() == "t" else tr[var], dtype=float)
    fin = np.isfinite(y)
    lo, hi = (float(np.min(y[fin])), float(np.max(y[fin]))) if fin.any() else (-1.0, 1.0)
    if hi - lo <= 1e-6 * max(abs(hi), abs(lo)):   # flat (data are single precision)
        lo, hi = lo - 1, hi + 1
    if var.lower() == "t":
        return lo, hi
    m = pad * (hi - lo)
    return lo - m, hi + m


def xpp_span(tr, var, pad=0.05):
    """span() for a window XPP draws: XPP cannot make a window narrower
    than about 1e-12, so a column with a smaller range gets +-1."""
    lo, hi = span(tr, var, pad)
    if hi - lo < 1e-12:
        mid = 0.5 * (lo + hi)
        return mid - 1, mid + 1
    return lo, hi


def plot_all_vs_t(tr, names, max_panels=16, title=None):
    """One small panel per variable against t (at most max_panels)."""
    import matplotlib.pyplot as plt
    shown = list(names)[:max_panels]
    n = len(shown)
    cols = 1 if n == 1 else 2 if n <= 4 else 3 if n <= 9 else 4
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.2 * rows + 0.6), squeeze=False)
    for ax, v in zip(axs.flat, shown):
        ax.plot(tr.t, tr[v], "k", lw=0.9)
        ax.set_ylabel(v)
        ax.set_xlim(*span(tr, "t"))
        ax.set_ylim(*span(tr, v))
    for ax in axs[-1]:
        ax.set_xlabel("t")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    more = len(names) - n
    fig.suptitle((title or "all variables against t") + (f"  (first {n} of {len(names)})" if more > 0 else ""))
    fig.tight_layout()
    return fig


def plot_projection(tr, x, y, ax=None, title=None):
    """y against x (a phase-space projection) from XPP's data."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.4))
    else:
        fig = ax.figure
    ax.plot(tr[x], tr[y], "k", lw=0.8)
    ax.plot(tr[x][:1], tr[y][:1], "o", mfc="w", mec="k", ms=5)
    ax.set_xlim(*span(tr, x))
    ax.set_ylim(*span(tr, y))
    ax.set(xlabel=x, ylabel=y, title=title or f"{y} vs {x}")
    return fig, ax


def plot_3d(tr, x, y, z, title=None):
    """Trajectory in (x, y, z) from XPP's data."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5.6, 5))
    ax = fig.add_subplot(projection="3d")
    ax.plot(tr[x], tr[y], tr[z], "k", lw=0.6)
    ax.set(xlabel=x, ylabel=y, zlabel=z, title=title or f"({x}, {y}, {z})")
    return fig, ax


def xpp_graph2d(m, x, y, path, xlim=None, ylim=None, tr=None):
    """XPP's own 2D graph of y against x (either may be t), as SVG."""
    tr = tr if tr is not None else m.data
    xlo, xhi = xlim or xpp_span(tr, x)
    ylo, yhi = ylim or xpp_span(tr, y)
    m.view(x=x, y=y, xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi)
    m.buttons.graphic_stuff.remove_all()
    m.press("r")
    m.buttons.graphic_stuff.svg(str(Path(path).resolve()))
    return Path(path)


def xpp_graph3d(m, x, y, z, path, tr=None):
    """XPP's own 3D graph of (x, y, z), as SVG (Viewaxes > 3D)."""
    tr = tr if tr is not None else m.data
    (x0, x1), (y0, y1), (z0, z1) = xpp_span(tr, x), xpp_span(tr, y), xpp_span(tr, z)
    d = {"X-axis": x, "Y-axis": y, "Z-axis": z, "Xmin": x0, "Xmax": x1, "Ymin": y0, "Ymax": y1,
         "Zmin": z0, "Zmax": z1, "XLo": -2, "XHi": 2, "YLo": -2, "YHi": 2}
    m._press_main(["v", "3"], (d,))
    m.buttons.graphic_stuff.remove_all()
    m.press("r")
    m.buttons.graphic_stuff.svg(str(Path(path).resolve()))
    return Path(path)


def compare_curve(tr, cols, svg_path, per_axis=None):
    """Does the curve XPP drew in svg_path match our data?

    cols: the data columns of the graph ("t" allowed).  per_axis: for a 2D
    graph the column of each screen axis ([x, y]); each screen coordinate
    is then fitted as a*value + b.  Without it (3D), each screen coordinate
    is fitted as a linear function of all the columns (a parallel
    projection).  Returns points, largest misfit in pixels, ok (<= 1.5 px
    and as many points as rows)."""
    px = svg_curve(svg_path)
    px = px[np.isfinite(px).all(axis=1)]
    col = lambda c: np.asarray(tr.t if c.lower() == "t" else tr[c], dtype=float)  # noqa: E731
    D = np.column_stack([col(c) for c in cols])
    keep = np.isfinite(D).all(axis=1)
    D = D[keep]
    out = {"xpp_points": int(len(px)), "data_points": int(len(D))}
    n = min(len(px), len(D))
    if n < 3:
        out.update(ok=False, why="too few points")
        return out
    P, D = px[:n], D[:n]
    errs = []
    for k in range(2):
        if per_axis:
            A = np.column_stack([D[:, cols.index(per_axis[k])], np.ones(n)])
        else:
            A = np.column_stack([D, np.ones(n)])
        # fit on the mean of the data at each pixel value, so that a curve
        # that spends most of its time in a narrow band does not set the
        # scale alone (its far points would then be off by several pixels)
        lev, inv = np.unique(P[:, k], return_inverse=True)
        cnt = np.bincount(inv)
        Am = np.column_stack([np.bincount(inv, weights=A[:, j]) / cnt for j in range(A.shape[1])])
        coef, *_ = np.linalg.lstsq(Am, lev, rcond=None)
        e1 = np.abs(A @ coef - P[:, k])
        # with few pixel levels that fit is underdetermined: also try the
        # plain fit on every point and keep the better one
        coef, *_ = np.linalg.lstsq(A, P[:, k], rcond=None)
        e2 = np.abs(A @ coef - P[:, k])
        errs.append(e1 if e1.max() <= e2.max() else e2)
    err = np.maximum(errs[0], errs[1])
    out.update(max_px=float(err.max()), mean_px=float(err.mean()),
               ok=bool(err.max() <= 1.5 and len(px) == len(D)))
    if len(px) != len(D):
        out["why"] = "XPP drew a different number of points"
    return out
