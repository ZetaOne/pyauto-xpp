"""
Write an XPP .ode file from Python, one statement at a time.

    from pyxpp.builder import ModelBuilder
    mb = (ModelBuilder("lecar")
          .parameter(iapp=0.0, phi=0.333)
          .function("minf(v)", ".5*(1+tanh((v-v1)/v2))")
          .ode("v", "iapp+gl*(vl-v)+gk*w*(vk-v)-ica")
          .ode("w", "lamn(v)*(ninf(v)-w)")
          .option(total=30, dt=0.05, xp="v", yp="w"))
    path = mb.write("lecar.ode")
    m = mb.load()                       # the same, loaded into XPP

Statements are written in the order the methods are called, because order
matters in an .ode file (fixed quantities are evaluated top to bottom, and
`%[i..j]` blocks and `set` lines refer to what came before).

Names may use XPP's array notation, e.g. ode("x[1..10]", "-x[j]+p").
Names that are not valid Python keywords (arrays, `x(0)`) are passed as a
dict: parameter({"a[1..3]": 0}).

Anything the methods do not cover can be written with raw(); a model that
needs raw() is a gap in this builder.
"""
from __future__ import annotations

import re

from pathlib import Path
from typing import Iterable, Mapping


def _fmt(v) -> str | None:
    """A value as XPP should read it (None: the name alone, XPP's 0)."""
    if v is None:
        return None
    import numbers
    if isinstance(v, bool) or type(v).__name__ == "bool_":
        return "1" if v else "0"
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real):            # also numpy floats (repr would give "np.float64(..)")
        return repr(float(v))
    return str(v)


def _pairs(values, kw: Mapping) -> list[tuple[str, str]]:
    """(name, value) pairs from a dict or a list of pairs, then keywords."""
    out = []
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        values = list(values)
        if any(not isinstance(p, tuple) or len(p) != 2 for p in values):
            raise TypeError("expected a dict or a list of (name, value) pairs")
        out.extend((str(k), _fmt(v)) for k, v in values)
        values = {}
    for src in values, kw:
        for k, v in src.items():
            out.append((str(k), _fmt(v)))
    return out


class ModelBuilder:
    """Build an .ode file statement by statement (see the module docstring)."""

    def __init__(self, name: str = "model", outdir: str | Path | None = None):
        """`outdir`: folder for the .ode file that write()/load() make when
        given no path, and (for load()) XPP's working folder, where its .set
        files, SVGs, AUTO files and exported data go.  Created if missing.
        Default: the .ode file in the current folder and a fresh temporary
        working folder per model."""
        self.name = name
        self.outdir = Path(outdir) if outdir is not None else None
        self.lines: list[str] = []
        self.notes_text: str | None = None
        self.raw_count = 0

    # -- plain text -------------------------------------------------------
    def comment(self, text: str = "") -> "ModelBuilder":
        """`# text` (one line per line of `text`)."""
        for ln in str(text).split("\n"):
            self.lines.append(f"# {ln}".rstrip() if ln else "#")
        return self

    def blank(self) -> "ModelBuilder":
        self.lines.append("")
        return self

    def raw(self, line: str) -> "ModelBuilder":
        """A line written exactly as given (not covered by the builder)."""
        self.lines.extend(str(line).split("\n"))
        self.raw_count += 1
        return self

    def tutorial(self, text: str) -> "ModelBuilder":
        """`" text`: a line of XPP's tutorial window; `{a=1,total=100}` in the
        text becomes a clickable action that sets those values."""
        self.lines.append(f'" {text}'.rstrip())
        return self

    def notes(self, text: str) -> "ModelBuilder":
        """Free text after `done` (XPP ignores it)."""
        self.notes_text = text
        return self

    # -- quantities -------------------------------------------------------
    def _decl(self, kw_word, values, kw, sep=","):
        pairs = _pairs(values, kw)
        if pairs:
            self.lines.append(f"{kw_word} " + sep.join(k if v is None else f"{k}={v}" for k, v in pairs))
        return self

    def parameter(self, values: Mapping | None = None, /, **kw) -> "ModelBuilder":
        """`par a=1,b=2` (one line per call)."""
        return self._decl("par", values, kw)

    def initial(self, values: Mapping | None = None, /, **kw) -> "ModelBuilder":
        """`init x=1,y=0` (numbers; for formulas use initial_value)."""
        return self._decl("init", values, kw)

    def initial_value(self, name: str, expr) -> "ModelBuilder":
        """`x(0)=expr`; also `x[1..5](0)=ran(1)`."""
        self.lines.append(f"{name}(0)={_fmt(expr)}")
        return self

    def number(self, values: Mapping | None = None, /, **kw) -> "ModelBuilder":
        """`number n=10` (a fixed number, not a parameter)."""
        return self._decl("number", values, kw)

    def derived(self, name: str, expr: str) -> "ModelBuilder":
        """`!name=expr`: a number computed from parameters, updated when they change."""
        self.lines.append(f"!{name}={expr}")
        return self

    def function(self, signature: str, expr: str) -> "ModelBuilder":
        """`f(x,y)=expr`."""
        self.lines.append(f"{signature}={expr}")
        return self

    def fixed(self, name: str, expr: str) -> "ModelBuilder":
        """`name=expr`: a fixed (temporary) quantity."""
        self.lines.append(f"{name}={expr}")
        return self

    # -- equations --------------------------------------------------------
    def ode(self, var: str, rhs: str) -> "ModelBuilder":
        """`var'=rhs`."""
        self.lines.append(f"{var}'={rhs}")
        return self

    def map(self, var: str, rhs: str) -> "ModelBuilder":
        """`var(t+1)=rhs` (difference equation; use option(meth="discrete"))."""
        self.lines.append(f"{var}(t+1)={rhs}")
        return self

    def volterra(self, var: str, rhs: str) -> "ModelBuilder":
        """`volt var=rhs` (integral equation; rhs uses int{...})."""
        self.lines.append(f"volt {var}={rhs}")
        return self

    def aux(self, name: str, expr: str) -> "ModelBuilder":
        """`aux name=expr` (a quantity saved with the data)."""
        self.lines.append(f"aux {name}={expr}")
        return self

    def algebraic(self, expr: str) -> "ModelBuilder":
        """`0=expr` (an algebraic condition of a DAE)."""
        self.lines.append(f"0={expr}")
        return self

    def solve(self, var: str, guess=None) -> "ModelBuilder":
        """`solv var=guess` (an algebraic variable of a DAE; no guess: 0)."""
        self.lines.append(f"solv {var}" if guess is None else f"solv {var}={_fmt(guess)}")
        return self

    def markov(self, var: str, rates: Iterable[Iterable[str]]) -> "ModelBuilder":
        """`markov var n` followed by the n x n matrix of transition rates."""
        rows = [list(r) for r in rates]
        self.lines.append(f"markov {var} {len(rows)}")
        for r in rows:
            self.lines.append(" ".join("{" + str(x) + "}" for x in r))
        return self

    def wiener(self, *names: str) -> "ModelBuilder":
        """`wiener w1,w2` (normal noise, scaled by sqrt(dt))."""
        self.lines.append("wiener " + ",".join(names))
        return self

    def table(self, name: str, file: str | None = None, *, n=None, lo=None, hi=None,
              expr: str | None = None) -> "ModelBuilder":
        """`table name file` or `table name % n lo hi expr`."""
        if file is not None:
            self.lines.append(f"table {name} {file}")
        else:
            self.lines.append(f"table {name} % {_fmt(n)} {_fmt(lo)} {_fmt(hi)} {expr}")
        return self

    def special(self, name: str, expr: str) -> "ModelBuilder":
        """`special name=expr` (conv, sparse, gill, mmult, ... )."""
        self.lines.append(f"special {name}={expr}")
        return self

    def global_event(self, sign: int, condition: str, actions) -> "ModelBuilder":
        """`global sign condition {a=1;b=2}`.  `actions`: dict or "a=1;b=2"."""
        if isinstance(actions, Mapping):
            actions = ";".join(f"{k}={_fmt(v)}" for k, v in actions.items())
        self.lines.append(f"global {sign} {condition} {{{actions}}}")
        return self

    def boundary(self, expr: str) -> "ModelBuilder":
        """`bndry expr` (`x` = value at the left end, `x'` = at the right end)."""
        self.lines.append(f"bndry {expr}")
        return self

    def export(self, inputs: Iterable[str], outputs: Iterable[str] = ()) -> "ModelBuilder":
        """`export {a,b} {c}` (pass values to a C library loaded with dll_lib)."""
        line = "export {" + ",".join(inputs) + "}"
        outputs = list(outputs)
        if outputs:
            line += " {" + ",".join(outputs) + "}"
        self.lines.append(line)
        return self

    def only(self, *names: str) -> "ModelBuilder":
        """`only x,y` (save only these columns in silent mode)."""
        self.lines.append("only " + ",".join(names))
        return self

    def begin_block(self, index_range: str) -> "ModelBuilder":
        """`%[lo..hi]`: the statements up to end_block() are repeated for
        j = lo..hi (write them with [j], [j+1], ...)."""
        self.lines.append(f"%[{index_range}]")
        return self

    def end_block(self) -> "ModelBuilder":
        """`%`: closes begin_block()."""
        self.lines.append("%")
        return self

    # -- settings ---------------------------------------------------------
    def option(self, values: Mapping | None = None, /, **kw) -> "ModelBuilder":
        """`@ total=100,dt=.05,xp=v,yp=w` (any @ option, incl. AUTO's).

        meth= is translated to XPP's own name: XPP reads only the first letter
        of the method, so "dopri5" or "dormprince" would silently become
        "discrete".  Unknown method names raise ValueError."""
        from .model import canonical_method
        def fix(d):
            return {k: (canonical_method(v) if str(k).lower() in ("meth", "method") else v)
                    for k, v in d.items()}
        return self._decl("@", fix(dict(values)) if values else None, fix(kw))

    def parameter_set(self, name: str, values: Mapping | None = None, /, **kw) -> "ModelBuilder":
        """`set name {a=1,total=200}`: a named set chosen from File > Get par set."""
        body = ",".join(f"{k}={v}" for k, v in _pairs(values, kw))
        self.lines.append(f"set {name} {{{body}}}")
        return self

    def options(self) -> dict:
        """The @ options written so far, {lower-case key: value text}."""
        d = {}
        for ln in self.lines:
            s = ln.strip()
            if not s.startswith("@"):
                continue
            text = re.sub(r"\s*=\s*", "=", s[1:].strip())
            for tok in re.split(r"[,\s]+", text):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    d[k.lower()] = v
        return d

    # -- output -----------------------------------------------------------
    def render(self) -> str:
        out = list(self.lines) + ["done"]
        if self.notes_text is not None:
            out.append(self.notes_text)
        return "\n".join(out) + "\n"

    def write(self, path: str | Path | None = None) -> Path:
        """Write the .ode file: to `path`, or `<outdir>/<name>.ode`, or
        `<name>.ode` in the current folder.  Missing folders are created."""
        if path is None:
            path = (self.outdir or Path(".")) / (self.name + ".ode")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())
        return path

    def load(self, path: str | Path | None = None, **kw):
        """Write the file and load it into XPP (returns a pyxpp Model).
        Keywords go to pyxpp.load; with `outdir` set, `workdir` defaults to it."""
        from .model import load
        if self.outdir is not None:
            kw.setdefault("workdir", self.outdir)
        return load(self.write(path), **kw)

    def __repr__(self):
        return f"<ModelBuilder {self.name}: {len(self.lines)} lines>"
